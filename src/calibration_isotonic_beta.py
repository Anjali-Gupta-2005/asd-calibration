import itertools
import numpy as np
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from config import COHORTS, ALL_MODELS, N_FOLDS
from src.utils import load_probs, save_calibrated_probs, set_seed

EPS = 1e-6


def isotonic_scale(cohort, model, fold):
    calib_probs, calib_labels = load_probs(cohort, model, fold, "calib")
    test_probs, _ = load_probs(cohort, model, fold, "test")

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(calib_probs, calib_labels)
    calibrated = ir.predict(test_probs)

    save_calibrated_probs(cohort, model, "isotonic", fold, calibrated)


def beta_scale(cohort, model, fold):
    calib_probs, calib_labels = load_probs(cohort, model, fold, "calib")
    test_probs, _ = load_probs(cohort, model, fold, "test")

    # betacal needs both classes present in the calibration slice, and can
    # be numerically unstable on very small/degenerate calib sets (this is
    # exactly the adolescent-cohort risk flagged in Krati's Platt/Temperature
    # script - small calib slices are hard for any parametric calibrator)
    if len(np.unique(calib_labels)) < 2:
        raise ValueError(
            f"beta_scale: {cohort}/{model}/fold{fold} calib slice has only "
            f"one class present - cannot fit a calibrator on it."
        )

    bc = BetaCalibration(parameters="abm")
    bc.fit(calib_probs, calib_labels)
    calibrated = bc.predict(test_probs)
    calibrated = np.clip(calibrated, 0.0, 1.0)

    save_calibrated_probs(cohort, model, "beta", fold, calibrated)


if __name__ == "__main__":
    set_seed()
    for cohort, model, fold in itertools.product(COHORTS, ALL_MODELS, range(N_FOLDS)):
        print(f"Isotonic + Beta: {cohort} / {model} / fold{fold}")
        isotonic_scale(cohort, model, fold)
        beta_scale(cohort, model, fold)
    print("Done. Isotonic + Beta applied to all 9 models x 4 cohorts x 3 folds.")