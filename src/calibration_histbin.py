"""
calibration_histbin.py
Krati's Phase 3, Day 4 script.

Implements Histogram Binning calibration:
  1. Bin the CALIBRATION-slice probabilities into N equal-width bins.
  2. For each bin, compute the observed positive-class frequency
     (using calibration-slice labels).
  3. Map each TEST-slice probability to its bin's observed frequency.

Empty-bin fallback: small cohorts (adolescent's calib-slice is ~21 rows
spread across 10 bins) will frequently produce empty bins. Empty bins are
filled by linear interpolation from the nearest non-empty bins; if every
bin is empty (degenerate case) we fall back to the calibration slice's
overall base rate.

Follows the exact same module-style pattern as calibration_isotonic_beta.py.

Run with: python -m src.calibration_histbin
(module-style — see calibration_platt_temp_hist.py's sibling files for why:
 this file uses `from config import ...` / `from src.utils import ...`,
 which requires the -m invocation from the project root.)
"""
import itertools
import numpy as np

from config import COHORTS, ALL_MODELS, N_FOLDS, ECE_N_BINS
from src.utils import load_probs, save_calibrated_probs, set_seed


def histogram_binning(cohort, model, fold, n_bins=ECE_N_BINS):
    calib_probs, calib_labels = load_probs(cohort, model, fold, "calib")
    test_probs, _ = load_probs(cohort, model, fold, "test")

    calib_probs = np.asarray(calib_probs, dtype=float)
    calib_labels = np.asarray(calib_labels, dtype=float)
    test_probs = np.asarray(test_probs, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Use only the interior edges for digitize so probabilities of exactly
    # 0.0 and 1.0 land in bin 0 and bin (n_bins - 1) respectively, not an
    # out-of-range bin.
    interior_edges = bin_edges[1:-1]

    bin_freq = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    calib_bin_idx = np.digitize(calib_probs, interior_edges, right=False)
    for i in range(n_bins):
        mask = calib_bin_idx == i
        bin_count[i] = mask.sum()
        if bin_count[i] > 0:
            bin_freq[i] = calib_labels[mask].mean()

    base_rate = calib_labels.mean() if len(calib_labels) > 0 else 0.5

    if np.isnan(bin_freq).all():
        # No calibration data at all fell into any bin (shouldn't happen
        # in practice, but guard against it) — fall back entirely to base rate.
        bin_freq[:] = base_rate
    elif np.isnan(bin_freq).any():
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        known = ~np.isnan(bin_freq)
        bin_freq[~known] = np.interp(
            bin_centers[~known], bin_centers[known], bin_freq[known]
        )

    test_bin_idx = np.digitize(test_probs, interior_edges, right=False)
    calibrated = bin_freq[test_bin_idx]

    save_calibrated_probs(cohort, model, "histbin", fold, calibrated)

    return bin_count  # useful for the verification script / empty-bin diagnostics


if __name__ == "__main__":
    set_seed()
    for cohort, model, fold in itertools.product(COHORTS, ALL_MODELS, range(N_FOLDS)):
        print(f"Histogram Binning: {cohort} / {model} / fold{fold}")
        counts = histogram_binning(cohort, model, fold)
        n_empty = int((counts == 0).sum())
        if n_empty > 0:
            print(f"  -> {n_empty}/{ECE_N_BINS} calibration bins were empty (interpolated)")
    print("Done. Histogram Binning applied to all 9 models x 4 cohorts x 3 folds.")

# Expect the adolescent cohort to show the most empty-bin warnings above —
# its ~21-row calibration slice spread across 10 bins is exactly the small-n
# scenario flagged in the project handoff. This is itself a data point for
# the paper's Discussion: histogram binning's reliability degrades on the
# smallest cohort in a different (bin-sparsity) way than Platt scaling's
# parametric instability does on the same cohort.
