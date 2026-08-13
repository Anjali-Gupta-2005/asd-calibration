"""
preprocessing.py
Harmonizes the 4 raw ASD screening datasets (toddler, child, adolescent, adult)
into a common schema, removes leakage, imputes missing values, encodes
categoricals, and builds 3 repeated stratified 60/20/20 train/calib/test splits
per cohort.

Run with:  python src/preprocessing.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# lets this script be run as `python src/preprocessing.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# ---------------------------------------------------------------------------
# 1. Column-name mapping: raw column name -> canonical name
#    (UCI Adult / Child / Adolescent files all share one raw format;
#     the Kaggle Toddler file uses a different raw format.)
# ---------------------------------------------------------------------------

UCI_SCORE_MAP = {f"A{i}_Score": f"A{i}" for i in range(1, 11)}

# ucimlrepo sometimes preserves the original typo'd column names from the
# source files ("jundice", "austim") and sometimes the corrected metadata
# names ("jaundice", "autism") - check for either, whichever is present.
UCI_RENAME_CANDIDATES = {
    "sex": ["gender", "sex"],
    "jaundice": ["jundice", "jaundice"],
    "family_mem_with_ASD": ["austim", "autism", "family_pdd", "family_mem_with_ASD"],
    "country_of_res": ["contry_of_res", "country_of_res"],
    "class_asd": ["Class/ASD", "class_asd", "Class/ASD Traits", "class"],
}
UCI_DROP_COLS = ["used_app_before", "result", "age_desc", "relation"]


def _rename_first_match(df, canonical, candidates):
    for cand in candidates:
        if cand in df.columns and cand != canonical:
            return df.rename(columns={cand: canonical})
    return df

TODDLER_RENAME_MAP = {
    **{f"A{i}": f"A{i}" for i in range(1, 11)},
    "Age_Mons": "age_months",
    "Sex": "sex",
    "Ethnicity": "ethnicity",
    "Jaundice": "jaundice",
    "Family_mem_with_ASD": "family_mem_with_ASD",
    "Class/ASD Traits": "class_asd",   # trailing space stripped before matching
}
TODDLER_DROP_COLS = ["Case_No", "Qchat-10-Score", "Who completed the test"]


def load_raw(cohort):
    path = os.path.join(cfg.PATH_RAW, f"{cohort}_raw.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # real files have trailing spaces
    return df


def harmonize_columns(df, cohort):
    if cohort == "toddler":
        df = df.rename(columns=TODDLER_RENAME_MAP)
        df = df.drop(columns=[c for c in TODDLER_DROP_COLS if c in df.columns])
        df["age"] = pd.to_numeric(df["age_months"], errors="coerce") / 12.0
        df = df.drop(columns=["age_months"])
        df["country_of_res"] = np.nan  # toddler dataset never collected this
    else:
        df = df.rename(columns=UCI_SCORE_MAP)
        for canonical, candidates in UCI_RENAME_CANDIDATES.items():
            df = _rename_first_match(df, canonical, candidates)
        df = df.drop(columns=[c for c in UCI_DROP_COLS if c in df.columns])
    return df


def clean_missing_and_types(df):
    # UCI files use "?" as the missing-value marker
    df = df.replace("?", np.nan)

    # numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # known data-entry error in the UCI adult dataset: at least one row has
    # age=383, which is physically impossible. Treat implausible ages as
    # missing so they get imputed like any other missing value, rather than
    # silently corrupting scale-sensitive models (SVM, KNN).
    df.loc[df["age"] > 100, "age"] = np.nan

    df["age"] = df["age"].fillna(df["age"].median())

    # categorical -> mode
    for col in ["sex", "ethnicity", "country_of_res"]:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df


def collapse_rare_categories(df, col, top_n=8):
    """Keeps the top_n most frequent values in a categorical column and bins
    everything else into 'Other'. Prevents one-hot encoding from exploding
    into dozens of near-empty dummy columns (e.g. country_of_res has 50+
    distinct values in these datasets, most appearing only 1-5 times)."""
    if col not in df.columns:
        return df
    top_values = df[col].value_counts().nlargest(top_n).index
    df[col] = df[col].where(df[col].isin(top_values), other="Other")
    return df


def encode_columns(df):
    df["sex"] = df["sex"].astype(str).str.lower().map({"m": 1, "f": 0}).fillna(0).astype(int)

    for col in ["jaundice", "family_mem_with_ASD"]:
        df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

    df["class_asd"] = (
        df["class_asd"].astype(str).str.lower().map({"yes": 1, "no": 0})
    )
    if df["class_asd"].isna().any():
        raise ValueError("class_asd contains values other than yes/no - check raw label column")
    df["class_asd"] = df["class_asd"].astype(int)

    for i in range(1, 11):
        df[f"A{i}"] = pd.to_numeric(df[f"A{i}"], errors="coerce").fillna(0).astype(int)

    # collapse rare ethnicity/country values to keep the one-hot feature
    # space manageable, especially important for the small adolescent cohort
    df = collapse_rare_categories(df, "ethnicity", top_n=8)
    df = collapse_rare_categories(df, "country_of_res", top_n=8)

    # one-hot encode ethnicity / country_of_res
    cat_cols = [c for c in ["ethnicity", "country_of_res"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    return df


def make_splits(df, cohort):
    y = df["class_asd"].values
    n = len(df)
    idx_all = np.arange(n)

    for fold in range(cfg.N_FOLDS):
        seed = cfg.SEED + fold
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            idx_all, y, test_size=(cfg.CALIB_FRAC + cfg.TEST_FRAC),
            stratify=y, random_state=seed
        )
        calib_idx, test_idx = train_test_split(
            temp_idx, test_size=cfg.TEST_FRAC / (cfg.CALIB_FRAC + cfg.TEST_FRAC),
            stratify=y_temp, random_state=seed
        )

        assert len(set(train_idx) & set(calib_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(calib_idx) & set(test_idx)) == 0

        split_dict = {
            "train_idx": train_idx.tolist(),
            "calib_idx": calib_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        out_path = os.path.join(cfg.PATH_SPLITS, f"{cohort}_fold{fold}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(split_dict, f)

        print(f"  fold{fold}: train={len(train_idx)} calib={len(calib_idx)} test={len(test_idx)}")


def process_cohort(cohort):
    print(f"\nProcessing cohort: {cohort}")
    df = load_raw(cohort)
    df = harmonize_columns(df, cohort)
    df = clean_missing_and_types(df)
    df = encode_columns(df)

    out_path = os.path.join(cfg.PATH_PROCESSED, f"{cohort}_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"  saved {out_path}  shape={df.shape}")

    make_splits(df, cohort)
    return df


def main():
    os.makedirs(cfg.PATH_PROCESSED, exist_ok=True)
    os.makedirs(cfg.PATH_SPLITS, exist_ok=True)

    for cohort in cfg.COHORTS:
        process_cohort(cohort)

    print("\nAll 4 cohorts processed successfully.")


if __name__ == "__main__":
    main()