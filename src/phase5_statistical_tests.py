"""
phase5_statistical_tests.py
Phase 5 (Day 6). Anjali (lead) — covers all 5 of Anjali's Day 6 tasks:
  1. Per-cohort ranking of the 5 calibration methods by mean ECE.
  2. Kendall's W across the 4 cohort rankings -> headline answer to the RQ.
  3. Friedman test (method = treatment, cohort x model = block, ECE = outcome).
  4. Nemenyi post-hoc test if Friedman is significant.
  5. Repeat ranking + Kendall's W using Brier score and MCE as robustness checks.

Reuses the Kendall's W trick from Doc 3 (Coffie, Kim & Chen) — applied to
calibration-method rankings instead of feature rankings, per the project's
stated core novelty.

Outputs:
  results/kendall_w_report.json
  results/friedman_nemenyi_report.txt

Run with: python -m src.phase5_statistical_tests
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, chi2, studentized_range, rankdata

from config import COHORTS, ALL_MODELS, ALL_METHODS, PATH_RESULTS

ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1 & 2. Per-cohort method ranking + Kendall's W
# ---------------------------------------------------------------------------

def per_cohort_method_ranking(df, metric_col):
    """
    For each cohort, averages `metric_col` across the 9 models for each
    method, then ranks the 5 methods (rank 1 = best = lowest value).
    Returns a DataFrame: index=cohort, columns=methods, values=rank.
    Also returns the underlying mean-value table for reporting.
    """
    calib_df = df[df["method"] != "raw"]
    mean_table = calib_df.groupby(["cohort", "method"])[metric_col].mean().unstack()
    mean_table = mean_table.reindex(index=COHORTS, columns=ALL_METHODS)

    rank_table = mean_table.rank(axis=1, method="average", ascending=True)
    return rank_table, mean_table


def kendalls_w(rank_table):
    """
    Classic Kendall's coefficient of concordance, with tie correction.
    rank_table: DataFrame, rows = raters (cohorts), columns = items (methods),
    values = ranks (ties allowed, using average-rank convention).
    Returns dict with W, chi2 statistic, df, p-value.
    """
    ranks = rank_table.values.astype(float)
    k, n = ranks.shape  # k raters (cohorts), n items (methods)

    R_j = ranks.sum(axis=0)  # column sums of ranks
    Rbar = k * (n + 1) / 2
    S = np.sum((R_j - Rbar) ** 2)

    # Tie correction term: sum over each rater's tied groups of (t^3 - t)
    T = 0.0
    for i in range(k):
        _, counts = np.unique(ranks[i], return_counts=True)
        T += np.sum(counts ** 3 - counts)

    denom = k ** 2 * (n ** 3 - n) - k * T
    W = (12 * S) / denom if denom > 0 else float("nan")

    stat = k * (n - 1) * W
    df = n - 1
    p_value = 1 - chi2.cdf(stat, df)

    return {
        "W": float(W),
        "chi2_stat": float(stat),
        "df": int(df),
        "p_value": float(p_value),
        "significant_agreement": bool(p_value < ALPHA),
        "n_raters_cohorts": int(k),
        "n_items_methods": int(n),
    }


def interpret_w(w):
    if w >= 0.7:
        return "STRONG agreement across cohorts -> one method tends to dominate everywhere"
    elif w >= 0.4:
        return "MODERATE agreement -> partial consistency, some stage-dependence"
    else:
        return "WEAK agreement -> best method changes substantially by developmental stage"


# ---------------------------------------------------------------------------
# 3. Friedman test (global, cohort x model as blocks)
# ---------------------------------------------------------------------------

def friedman_test(df, metric_col):
    calib_df = df[df["method"] != "raw"].copy()
    pivot = calib_df.pivot_table(index=["cohort", "model"], columns="method", values=metric_col)
    pivot = pivot.dropna()  # only complete blocks (all 5 methods present)
    pivot = pivot[ALL_METHODS]  # fixed column order

    stat, p = friedmanchisquare(*[pivot[m].values for m in ALL_METHODS])
    return {
        "stat": float(stat),
        "p_value": float(p),
        "significant": bool(p < ALPHA),
        "n_blocks": int(len(pivot)),
        "methods": ALL_METHODS,
    }, pivot


# ---------------------------------------------------------------------------
# 4. Nemenyi post-hoc (manual implementation via studentized range distribution
# — avoids depending on scikit-posthocs, which may not be installed)
# ---------------------------------------------------------------------------

def nemenyi_posthoc(pivot):
    """
    pivot: DataFrame, rows = blocks (cohort,model), columns = methods, values = metric.
    Returns mean ranks per method, critical difference, and a pairwise
    significance matrix.
    """
    n_blocks, k = pivot.shape
    ranks = pivot.apply(lambda row: rankdata(row), axis=1, result_type="expand")
    ranks.columns = pivot.columns
    mean_ranks = ranks.mean(axis=0)

    q_alpha = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_blocks))

    methods = list(pivot.columns)
    sig_matrix = pd.DataFrame(False, index=methods, columns=methods)
    diff_matrix = pd.DataFrame(0.0, index=methods, columns=methods)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:
                continue
            diff = abs(mean_ranks[m1] - mean_ranks[m2])
            diff_matrix.loc[m1, m2] = diff
            diff_matrix.loc[m2, m1] = diff
            is_sig = diff > cd
            sig_matrix.loc[m1, m2] = is_sig
            sig_matrix.loc[m2, m1] = is_sig

    return mean_ranks, cd, diff_matrix, sig_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_metric(df, metric_col, label):
    print(f"\n{'=' * 70}\nMETRIC: {label}\n{'=' * 70}")

    rank_table, mean_table = per_cohort_method_ranking(df, metric_col)
    print(f"\nMean {label} per (cohort, method):")
    print(mean_table.round(4).to_string())
    print(f"\nRank per (cohort, method) [1 = best]:")
    print(rank_table.round(1).to_string())

    w_result = kendalls_w(rank_table)
    print(f"\nKendall's W = {w_result['W']:.4f}  "
          f"(chi2={w_result['chi2_stat']:.3f}, df={w_result['df']}, p={w_result['p_value']:.4f})")
    print(f"Interpretation: {interpret_w(w_result['W'])}")

    return w_result, rank_table, mean_table


def main():
    os.makedirs(PATH_RESULTS, exist_ok=True)
    csv_path = os.path.join(PATH_RESULTS, "master_calibration_table.csv")
    df = pd.read_csv(csv_path)

    kendall_report = {}
    text_report_lines = []

    # --- Primary metric: ECE (full analysis: ranking + W + Friedman + Nemenyi) ---
    w_ece, rank_ece, mean_ece = run_for_metric(df, "ece_mean", "ECE (primary metric)")
    kendall_report["ece"] = {
        "kendalls_w": w_ece,
        "per_cohort_ranks": rank_ece.round(2).to_dict(orient="index"),
        "per_cohort_mean_values": mean_ece.round(4).to_dict(orient="index"),
    }

    print(f"\n{'=' * 70}\nFRIEDMAN TEST (ECE, cohort x model blocks, n=36)\n{'=' * 70}")
    friedman_result, pivot_ece = friedman_test(df, "ece_mean")
    print(f"Friedman chi2 = {friedman_result['stat']:.4f}, p = {friedman_result['p_value']:.6f}, "
          f"n_blocks = {friedman_result['n_blocks']}")

    text_report_lines.append("=" * 70)
    text_report_lines.append("FRIEDMAN TEST (ECE, cohort x model as blocks)")
    text_report_lines.append("=" * 70)
    text_report_lines.append(f"chi2 = {friedman_result['stat']:.4f}")
    text_report_lines.append(f"p-value = {friedman_result['p_value']:.6f}")
    text_report_lines.append(f"n_blocks = {friedman_result['n_blocks']}")
    text_report_lines.append(f"Significant at alpha={ALPHA}: {friedman_result['significant']}")

    if friedman_result["significant"]:
        print("\nFriedman significant -> running Nemenyi post-hoc...")
        mean_ranks, cd, diff_matrix, sig_matrix = nemenyi_posthoc(pivot_ece)

        print(f"\nMean ranks per method (lower = better):\n{mean_ranks.round(3).to_string()}")
        print(f"\nCritical difference (alpha={ALPHA}): {cd:.4f}")
        print(f"\nPairwise mean-rank differences:\n{diff_matrix.round(3).to_string()}")
        print(f"\nSignificant pairwise differences (True = differ significantly):\n{sig_matrix.to_string()}")

        text_report_lines.append("\n" + "-" * 70)
        text_report_lines.append("NEMENYI POST-HOC")
        text_report_lines.append("-" * 70)
        text_report_lines.append(f"Mean ranks per method (lower = better):\n{mean_ranks.round(3).to_string()}")
        text_report_lines.append(f"\nCritical difference (alpha={ALPHA}): {cd:.4f}")
        text_report_lines.append(f"\nPairwise mean-rank differences:\n{diff_matrix.round(3).to_string()}")
        text_report_lines.append(f"\nSignificant pairwise differences:\n{sig_matrix.to_string()}")

        sig_pairs = [(m1, m2) for m1 in sig_matrix.index for m2 in sig_matrix.columns
                     if m1 < m2 and sig_matrix.loc[m1, m2]]
        print(f"\nMethod pairs that differ significantly: {sig_pairs if sig_pairs else 'NONE'}")
        text_report_lines.append(f"\nMethod pairs that differ significantly: {sig_pairs if sig_pairs else 'NONE'}")
    else:
        print("\nFriedman NOT significant -> no Nemenyi post-hoc run "
              "(no significant overall difference among methods to localize).")
        text_report_lines.append("\nFriedman not significant - Nemenyi post-hoc skipped.")

    # --- Robustness checks: repeat ranking + Kendall's W for Brier and MCE ---
    w_brier, rank_brier, mean_brier = run_for_metric(df, "brier_mean", "Brier score (robustness check)")
    kendall_report["brier"] = {
        "kendalls_w": w_brier,
        "per_cohort_ranks": rank_brier.round(2).to_dict(orient="index"),
        "per_cohort_mean_values": mean_brier.round(4).to_dict(orient="index"),
    }

    w_mce, rank_mce, mean_mce = run_for_metric(df, "mce_mean", "MCE (robustness check)")
    kendall_report["mce"] = {
        "kendalls_w": w_mce,
        "per_cohort_ranks": rank_mce.round(2).to_dict(orient="index"),
        "per_cohort_mean_values": mean_mce.round(4).to_dict(orient="index"),
    }

    # --- Cross-metric consistency check ---
    print(f"\n{'=' * 70}\nCROSS-METRIC CONSISTENCY CHECK\n{'=' * 70}")
    print(f"Kendall's W -> ECE: {w_ece['W']:.4f} | Brier: {w_brier['W']:.4f} | MCE: {w_mce['W']:.4f}")
    ws = [w_ece["W"], w_brier["W"], w_mce["W"]]
    consistent = (max(ws) - min(ws)) < 0.2
    print(f"Consistent conclusion across all 3 metrics: {consistent} "
          f"(spread = {max(ws) - min(ws):.4f})")

    kendall_report["cross_metric_consistency"] = {
        "ece_w": w_ece["W"], "brier_w": w_brier["W"], "mce_w": w_mce["W"],
        "consistent": bool(consistent),
    }

    # --- Save outputs ---
    json_path = os.path.join(PATH_RESULTS, "kendall_w_report.json")
    with open(json_path, "w") as f:
        json.dump(kendall_report, f, indent=2)
    print(f"\nSaved -> {json_path}")

    txt_path = os.path.join(PATH_RESULTS, "friedman_nemenyi_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(text_report_lines))
    print(f"Saved -> {txt_path}")

    print("\nDone. Phase 5 complete.")


if __name__ == "__main__":
    main()
