import pandas as pd
from config import PATH_PROCESSED, COHORTS

for cohort in COHORTS:
    print(f"\n=== {cohort} ===")
    df = pd.read_csv(f"{PATH_PROCESSED}{cohort}_clean.csv")
    target = "class_asd"

    if target not in df.columns:
        print(f"  Target column '{target}' not found — check actual name.")
        continue

    # correlation of every numeric column with the target
    corr = df.corr(numeric_only=True)[target].drop(target).sort_values(key=abs, ascending=False)
    print(corr.head(10))


for cohort in COHORTS:
    df = pd.read_csv(f"{PATH_PROCESSED}{cohort}_clean.csv")
    a_cols = [f"A{i}" for i in range(1, 11)]
    a_cols = [c for c in a_cols if c in df.columns]

    df["_a_sum"] = df[a_cols].sum(axis=1)
    corr_with_sum = df["_a_sum"].corr(df["class_asd"])
    print(f"{cohort}: corr(sum of A1-A10, label) = {corr_with_sum:.4f}")

for cohort in COHORTS:
    df = pd.read_csv(f"{PATH_PROCESSED}{cohort}_clean.csv")
    numeric_cols = df.select_dtypes(include="number").columns
    print(f"\n=== {cohort} ===")
    print(df[numeric_cols].describe().loc[["min", "max"]])