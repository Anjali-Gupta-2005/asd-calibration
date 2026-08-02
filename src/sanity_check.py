"""
sanity_check.py
Run this after preprocessing.py, before every git commit/push of Phase 1
(or any later phase) output. Catches silent data-quality bugs that don't
throw errors but would quietly corrupt every downstream result.

Run with: python src/sanity_check.py
"""

import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

ISSUES = []


def check(condition, message):
    if condition:
        print(f"  OK   - {message}")
    else:
        print(f"  FAIL - {message}")
        ISSUES.append(message)


def check_processed_csvs():
    print("\n[1] Checking data/processed/*.csv")
    for cohort in cfg.COHORTS:
        path = os.path.join(cfg.PATH_PROCESSED, f"{cohort}_clean.csv")
        check(os.path.exists(path), f"{cohort}_clean.csv exists")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)

        check(df.isna().sum().sum() == 0, f"{cohort}: no missing values remain")
        check("class_asd" in df.columns, f"{cohort}: class_asd column present")
        if "class_asd" in df.columns:
            check(
                set(df["class_asd"].unique()) <= {0, 1},
                f"{cohort}: class_asd is strictly binary (0/1)",
            )
            pos_rate = df["class_asd"].mean()
            print(f"       -> {cohort} class balance: {pos_rate:.1%} positive, n={len(df)}")

        for i in range(1, 11):
            col = f"A{i}"
            if col in df.columns:
                check(
                    set(df[col].unique()) <= {0, 1},
                    f"{cohort}: {col} is strictly binary (0/1)",
                )


def check_splits():
    print("\n[2] Checking split_indices/*.pkl")
    for cohort in cfg.COHORTS:
        csv_path = os.path.join(cfg.PATH_PROCESSED, f"{cohort}_clean.csv")
        if not os.path.exists(csv_path):
            continue
        n_rows = len(pd.read_csv(csv_path))

        for fold in range(cfg.N_FOLDS):
            split_path = os.path.join(cfg.PATH_SPLITS, f"{cohort}_fold{fold}.pkl")
            check(os.path.exists(split_path), f"{cohort} fold{fold}: split file exists")
            if not os.path.exists(split_path):
                continue
            with open(split_path, "rb") as f:
                splits = pickle.load(f)

            train, calib, test = set(splits["train_idx"]), set(splits["calib_idx"]), set(splits["test_idx"])

            check(len(train & calib) == 0, f"{cohort} fold{fold}: train/calib do not overlap")
            check(len(train & test) == 0, f"{cohort} fold{fold}: train/test do not overlap")
            check(len(calib & test) == 0, f"{cohort} fold{fold}: calib/test do not overlap")
            check(
                len(train | calib | test) == n_rows,
                f"{cohort} fold{fold}: train+calib+test covers all {n_rows} rows",
            )


def check_schema_consistency():
    print("\n[3] Checking cross-cohort core-column consistency")
    required_core = [f"A{i}" for i in range(1, 11)] + [
        "age", "sex", "jaundice", "family_mem_with_ASD", "class_asd"
    ]
    for cohort in cfg.COHORTS:
        path = os.path.join(cfg.PATH_PROCESSED, f"{cohort}_clean.csv")
        if not os.path.exists(path):
            continue
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [c for c in required_core if c not in cols]
        check(len(missing) == 0, f"{cohort}: has all core columns {required_core if missing else ''}".split(" (")[0])
        if missing:
            print(f"       -> {cohort} is missing: {missing}")


def main():
    check_processed_csvs()
    check_splits()
    check_schema_consistency()

    print("\n" + "=" * 50)
    if ISSUES:
        print(f"{len(ISSUES)} ISSUE(S) FOUND - fix these before committing:")
        for issue in ISSUES:
            print(f"  - {issue}")
    else:
        print("All checks passed. Safe to commit and push.")


if __name__ == "__main__":
    main()