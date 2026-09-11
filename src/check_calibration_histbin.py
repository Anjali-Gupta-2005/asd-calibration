"""
check_calibration_histbin.py
Verifies that Histogram Binning actually improved calibration, by comparing
ECE and Brier score before (raw model output) vs after histbin calibration.
Mirrors check_calibration_isotonic_beta.py exactly, so results are directly
comparable across method-verification scripts.

Run this AFTER calibration_histbin.py.

Run with: python -m src.check_calibration_histbin
"""
import numpy as np
from config import COHORTS, ALL_MODELS, N_FOLDS, ECE_N_BINS
from src.utils import load_probs, load_calibrated_probs


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


def compute_brier(probs, labels):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    return np.mean((probs - labels) ** 2)


def main():
    print(f"{'cohort':<12}{'model':<10}{'raw_ECE':>9}{'histbin_ECE':>13}"
          f"{'raw_Brier':>11}{'histbin_Brier':>15}")
    print("-" * 75)

    improved = 0
    total = 0

    for cohort in COHORTS:
        for model in ALL_MODELS:
            raw_eces, hist_eces = [], []
            raw_briers, hist_briers = [], []

            for fold in range(N_FOLDS):
                raw_probs, labels = load_probs(cohort, model, fold, "test")
                hist_probs = load_calibrated_probs(cohort, model, "histbin", fold)

                raw_eces.append(compute_ece(raw_probs, labels))
                hist_eces.append(compute_ece(hist_probs, labels))

                raw_briers.append(compute_brier(raw_probs, labels))
                hist_briers.append(compute_brier(hist_probs, labels))

            raw_ece, hist_ece = np.mean(raw_eces), np.mean(hist_eces)
            raw_brier, hist_brier = np.mean(raw_briers), np.mean(hist_briers)

            print(f"{cohort:<12}{model:<10}{raw_ece:>9.4f}{hist_ece:>13.4f}"
                  f"{raw_brier:>11.4f}{hist_brier:>15.4f}")

            total += 1
            if hist_ece < raw_ece:
                improved += 1

    print("-" * 75)
    print(f"Histogram Binning improved ECE over raw in {improved}/{total} (cohort, model) combinations")


if __name__ == "__main__":
    main()
