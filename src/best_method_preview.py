"""
best_method_preview.py
Quick, non-statistical preview of Phase 5: for each (cohort, model), which
calibration method has the lowest mean ECE? Reads straight from
results/master_calibration_table.csv (Phase 4 output) — no new computation.

This is NOT the actual research-question test (that's Kendall's W /
Friedman / Nemenyi in Phase 5) — it's a plain-English look at the same
question before running the formal stats, so you can sanity-check whether
the pattern already looks cohort-dependent or already looks consistent.

Run with: python -m src.best_method_preview
"""
import os
import pandas as pd

from config import COHORTS, ALL_MODELS, ALL_METHODS, PATH_RESULTS


def main():
    csv_path = os.path.join(PATH_RESULTS, "master_calibration_table.csv")
    df = pd.read_csv(csv_path)

    # Only real calibration methods compete for "best" — raw is the baseline
    # being improved upon, not a candidate winner.
    calib_df = df[df["method"] != "raw"].copy()

    rows = []
    for cohort in COHORTS:
        for model in ALL_MODELS:
            sub = calib_df[(calib_df["cohort"] == cohort) & (calib_df["model"] == model)]
            if sub.empty:
                continue
            best = sub.loc[sub["ece_mean"].idxmin()]
            raw_row = df[(df["cohort"] == cohort) & (df["model"] == model) & (df["method"] == "raw")]
            raw_ece = raw_row["ece_mean"].values[0] if not raw_row.empty else float("nan")
            rows.append({
                "cohort": cohort,
                "model": model,
                "best_method": best["method"],
                "best_ece": round(best["ece_mean"], 4),
                "raw_ece": round(raw_ece, 4),
            })

    result = pd.DataFrame(rows)

    print("=== Best calibration method per (cohort, model) ===\n")
    print(result.to_string(index=False))

    print("\n=== Win count per method, by cohort ===")
    tally = result.groupby(["cohort", "best_method"]).size().unstack(fill_value=0)
    tally = tally.reindex(columns=ALL_METHODS, fill_value=0)
    print(tally.to_string())

    print("\n=== Win count per method, overall (36 combinations) ===")
    overall = result["best_method"].value_counts().reindex(ALL_METHODS, fill_value=0)
    print(overall.to_string())

    print("\n=== Does the winning method stay the same across cohorts, per model? ===")
    for model in ALL_MODELS:
        winners = result[result["model"] == model].set_index("cohort")["best_method"]
        n_unique = winners.nunique()
        status = "SAME winner in all 4 cohorts" if n_unique == 1 else f"CHANGES ({n_unique} different winners)"
        print(f"  {model:<10} {dict(winners)}  -> {status}")

    n_same = sum(
        result[result["model"] == m].set_index("cohort")["best_method"].nunique() == 1
        for m in ALL_MODELS
    )
    print(f"\n{n_same}/{len(ALL_MODELS)} models have ONE method winning across all 4 cohorts; "
          f"{len(ALL_MODELS) - n_same}/{len(ALL_MODELS)} have the winner change by cohort.")
    print("(This is the plain-English version of what Kendall's W will test formally in Phase 5.)")


if __name__ == "__main__":
    main()
