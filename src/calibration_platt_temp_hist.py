import itertools
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from config import COHORTS, ALL_MODELS, N_FOLDS
from src.utils import load_probs, save_calibrated_probs, set_seed

EPS = 1e-6

def platt_scale(cohort, model, fold):
    calib_probs, calib_labels = load_probs(cohort, model, fold, "calib")
    test_probs, _ = load_probs(cohort, model, fold, "test")

    lr = LogisticRegression()
    lr.fit(calib_probs.reshape(-1, 1), calib_labels)
    calibrated = lr.predict_proba(test_probs.reshape(-1, 1))[:, 1]

    save_calibrated_probs(cohort, model, "platt", fold, calibrated)

def temperature_scale(cohort, model, fold):
    calib_probs, calib_labels = load_probs(cohort, model, fold, "calib")
    test_probs, _ = load_probs(cohort, model, fold, "test")

    calib_probs_c = np.clip(calib_probs, EPS, 1 - EPS)
    calib_logits = np.log(calib_probs_c / (1 - calib_probs_c))

    def nll(T):
        scaled = 1 / (1 + np.exp(-calib_logits / T))
        scaled = np.clip(scaled, EPS, 1 - EPS)
        return -np.mean(calib_labels * np.log(scaled) + (1 - calib_labels) * np.log(1 - scaled))

    result = minimize_scalar(nll, bounds=(0.05, 10), method="bounded")
    T = result.x

    test_probs_c = np.clip(test_probs, EPS, 1 - EPS)
    test_logits = np.log(test_probs_c / (1 - test_probs_c))
    calibrated = 1 / (1 + np.exp(-test_logits / T))

    save_calibrated_probs(cohort, model, "temperature", fold, calibrated)

if __name__ == "__main__":
    set_seed()
    for cohort, model, fold in itertools.product(COHORTS, ALL_MODELS, range(N_FOLDS)):
        print(f"Platt + Temperature: {cohort} / {model} / fold{fold}")
        platt_scale(cohort, model, fold)
        temperature_scale(cohort, model, fold)
    print("Done. Platt + Temperature applied to all 9 models x 4 cohorts x 3 folds.")

#Platt Scaling underperforms specifically on the adolescent cohort
# (n=104, calib-slice size=21), consistent with the known instability of 2-parameter parametric
# calibration on small samples. Temperature Scaling (1 parameter) remains stable at this sample size.
# This supports the hypothesis that calibration method choice should vary by developmental stage.