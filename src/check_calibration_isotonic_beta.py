"""
check_calibration_isotonic_beta.py
Verifies that isotonic/beta calibration actually improved things, by
comparing ECE and Brier score before (raw model output) vs after
calibration. Run this after calibration_isotonic_beta.py.

Run with: python -m src.check_calibration_isotonic_beta
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
    print(f"{'cohort':<12}{'model':<10}{'raw_ECE':>9}{'iso_ECE':>9}{'beta_ECE':>9}"
          f"{'raw_Brier':>11}{'iso_Brier':>11}{'beta_Brier':>11}")
    print("-" * 85)

    improved_iso = 0
    improved_beta = 0
    total = 0

    for cohort in COHORTS:
        for model in ALL_MODELS:
            raw_eces, iso_eces, beta_eces = [], [], []
            raw_briers, iso_briers, beta_briers = [], [], []

            for fold in range(N_FOLDS):
                raw_probs, labels = load_probs(cohort, model, fold, "test")
                iso_probs = load_calibrated_probs(cohort, model, "isotonic", fold)
                beta_probs = load_calibrated_probs(cohort, model, "beta", fold)

                raw_eces.append(compute_ece(raw_probs, labels))
                iso_eces.append(compute_ece(iso_probs, labels))
                beta_eces.append(compute_ece(beta_probs, labels))

                raw_briers.append(compute_brier(raw_probs, labels))
                iso_briers.append(compute_brier(iso_probs, labels))
                beta_briers.append(compute_brier(beta_probs, labels))

            raw_ece, iso_ece, beta_ece = np.mean(raw_eces), np.mean(iso_eces), np.mean(beta_eces)
            raw_brier, iso_brier, beta_brier = np.mean(raw_briers), np.mean(iso_briers), np.mean(beta_briers)

            print(f"{cohort:<12}{model:<10}{raw_ece:>9.4f}{iso_ece:>9.4f}{beta_ece:>9.4f}"
                  f"{raw_brier:>11.4f}{iso_brier:>11.4f}{beta_brier:>11.4f}")

            total += 1
            if iso_ece < raw_ece:
                improved_iso += 1
            if beta_ece < raw_ece:
                improved_beta += 1

    print("-" * 85)
    print(f"Isotonic improved ECE over raw in {improved_iso}/{total} (cohort, model) combinations")
    print(f"Beta improved ECE over raw in {improved_beta}/{total} (cohort, model) combinations")


if __name__ == "__main__":
    main()