"""
evaluation.py
Phase 4 (Day 5). Anjali (lead) — since Krati is out, this also covers her
Day 5 task (reliability diagrams).

Computes ECE (10-bin), MCE, and Brier score for:
  - the RAW (uncalibrated) baseline, per (cohort, model)
  - all 5 calibration methods, per (cohort, model, method)
Averaged across the 3 folds, with std across folds also reported so
Phase 5's statistical tests have variance to work with.

Outputs:
  results/master_calibration_table.csv
  results/reliability_diagrams/{cohort}_{model}_{method}.png   (raw + 5 methods)

Run with: python -m src.evaluation
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe: never opens a GUI window
import matplotlib.pyplot as plt

from config import COHORTS, ALL_MODELS, ALL_METHODS, N_FOLDS, ECE_N_BINS, PATH_RESULTS
from src.utils import load_probs, load_calibrated_probs, set_seed

RELIABILITY_DIR = os.path.join(PATH_RESULTS, "reliability_diagrams")


# ---------------------------------------------------------------------------
# Metrics — same binning convention used throughout the project
# (check_calibration_isotonic_beta.py, check_calibration_histbin.py) so every
# number in this table is directly comparable to the earlier spot-checks.
# ---------------------------------------------------------------------------

def compute_ece(probs, labels, n_bins=ECE_N_BINS):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def compute_mce(probs, labels, n_bins=ECE_N_BINS):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        max_gap = max(max_gap, abs(acc - conf))
    return max_gap


def compute_brier(probs, labels):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    return float(np.mean((probs - labels) ** 2))


# ---------------------------------------------------------------------------
# Loading helper — treats "raw" as a pseudo-method so it flows through the
# same loop as the 5 real calibration methods
# ---------------------------------------------------------------------------

def get_probs_labels(cohort, model, method, fold):
    """Returns (probs, labels) for the test slice, for 'raw' or any real method."""
    if method == "raw":
        return load_probs(cohort, model, fold, "test")
    _, labels = load_probs(cohort, model, fold, "test")
    probs = load_calibrated_probs(cohort, model, method, fold)
    return probs, labels


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------

def build_master_table():
    rows = []
    methods_with_baseline = ["raw"] + ALL_METHODS

    for cohort in COHORTS:
        for model in ALL_MODELS:
            for method in methods_with_baseline:
                fold_ece, fold_mce, fold_brier = [], [], []
                missing = False

                for fold in range(N_FOLDS):
                    try:
                        probs, labels = get_probs_labels(cohort, model, method, fold)
                    except FileNotFoundError as e:
                        print(f"  MISSING: {cohort}/{model}/{method}/fold{fold} -> {e}")
                        missing = True
                        continue

                    fold_ece.append(compute_ece(probs, labels))
                    fold_mce.append(compute_mce(probs, labels))
                    fold_brier.append(compute_brier(probs, labels))

                if not fold_ece:
                    continue

                rows.append({
                    "cohort": cohort,
                    "model": model,
                    "method": method,
                    "ece_mean": np.mean(fold_ece),
                    "ece_std": np.std(fold_ece),
                    "mce_mean": np.mean(fold_mce),
                    "mce_std": np.std(fold_mce),
                    "brier_mean": np.mean(fold_brier),
                    "brier_std": np.std(fold_brier),
                    "n_folds_found": len(fold_ece),
                    "complete": (not missing) and len(fold_ece) == N_FOLDS,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reliability diagrams — pooled across the 3 folds for a stable per-bin
# estimate (especially needed for adolescent, where any single fold's
# test slice is only ~21 rows)
# ---------------------------------------------------------------------------

def plot_reliability(cohort, model, method, n_bins=ECE_N_BINS):
    all_probs, all_labels = [], []
    for fold in range(N_FOLDS):
        try:
            probs, labels = get_probs_labels(cohort, model, method, fold)
        except FileNotFoundError:
            return False
        all_probs.append(probs)
        all_labels.append(labels)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        bin_count[i] = mask.sum()
        if mask.sum() > 0:
            bin_acc[i] = labels[mask].mean()
            bin_conf[i] = probs[mask].mean()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    valid = ~np.isnan(bin_acc)
    ax.bar(bin_centers[valid], bin_acc[valid], width=1 / n_bins * 0.9,
           edgecolor="black", alpha=0.7, label="Observed frequency")
    ax.set_xlabel("Predicted probability (bin midpoint)")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title(f"{cohort} / {model} / {method}\n(n={len(probs)}, pooled across {N_FOLDS} folds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    os.makedirs(RELIABILITY_DIR, exist_ok=True)
    out_path = os.path.join(RELIABILITY_DIR, f"{cohort}_{model}_{method}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def build_all_reliability_diagrams():
    methods_with_baseline = ["raw"] + ALL_METHODS
    made, skipped = 0, 0
    for cohort in COHORTS:
        for model in ALL_MODELS:
            for method in methods_with_baseline:
                ok = plot_reliability(cohort, model, method)
                made += int(ok)
                skipped += int(not ok)
    print(f"Reliability diagrams: {made} created, {skipped} skipped (missing data)")


# ---------------------------------------------------------------------------
# End-of-day sanity check (matches the roadmap's Day 5 "confirm AdaBoost
# shows high ECE pre-calibration and check whether any method fixes it")
# ---------------------------------------------------------------------------

def print_adaboost_check(df):
    print("\n--- AdaBoost sanity check (raw vs. each method, mean ECE) ---")
    sub = df[df["model"] == "adaboost"].pivot(index="cohort", columns="method", values="ece_mean")
    cols = ["raw"] + [c for c in ALL_METHODS if c in sub.columns]
    print(sub[cols].round(4).to_string())


if __name__ == "__main__":
    set_seed()
    os.makedirs(PATH_RESULTS, exist_ok=True)

    print("Building master calibration table...")
    df = build_master_table()
    out_csv = os.path.join(PATH_RESULTS, "master_calibration_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows -> {out_csv}")

    n_expected = len(COHORTS) * len(ALL_MODELS) * (1 + len(ALL_METHODS))
    if len(df) != n_expected:
        print(f"  WARNING: expected {n_expected} rows (4 cohorts x 9 models x 6 methods-incl-raw), got {len(df)}")
    if not df["complete"].all():
        n_incomplete = (~df["complete"]).sum()
        print(f"  WARNING: {n_incomplete} rows built from fewer than {N_FOLDS} folds — check MISSING lines above")

    print_adaboost_check(df)

    print("\nBuilding reliability diagrams (this may take a minute)...")
    build_all_reliability_diagrams()

    print("\nDone. Phase 4 complete.")
