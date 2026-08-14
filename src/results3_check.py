import numpy as np
from sklearn.metrics import brier_score_loss
from config import COHORTS, ALL_MODELS, N_FOLDS
from src.utils import load_probs, load_calibrated_probs

print(f"{'Cohort':<12}{'Model':<10}{'Fold':<6}{'Raw':<10}{'Platt':<10}{'Temp':<10}")
for cohort in COHORTS:
    for model in ALL_MODELS:
        for fold in range(N_FOLDS):
            raw_probs, labels = load_probs(cohort, model, fold, "test")
            platt_probs = load_calibrated_probs(cohort, model, "platt", fold)
            temp_probs = load_calibrated_probs(cohort, model, "temperature", fold)

            raw_b = brier_score_loss(labels, raw_probs)
            platt_b = brier_score_loss(labels, platt_probs)
            temp_b = brier_score_loss(labels, temp_probs)

            print(f"{cohort:<12}{model:<10}{fold:<6}{raw_b:<10.4f}{platt_b:<10.4f}{temp_b:<10.4f}")
            