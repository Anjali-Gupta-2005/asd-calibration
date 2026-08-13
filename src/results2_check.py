import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from config import COHORTS, MODELS_CLASSICAL, N_FOLDS
from src.utils import load_probs

print(f"{'Cohort':<12}{'Model':<10}{'Fold':<6}{'Acc':<8}{'AUC':<8}{'Brier':<8}")
print("-" * 60)

for cohort in COHORTS:
    for model in MODELS_CLASSICAL:
        for fold in range(N_FOLDS):
            probs, labels = load_probs(cohort, model, fold, "test")
            preds = (probs >= 0.5).astype(int)

            acc = accuracy_score(labels, preds)
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = float("nan")  # happens if a fold has only one class present
            brier = brier_score_loss(labels, probs)

            print(f"{cohort:<12}{model:<10}{fold:<6}{acc:<8.3f}{auc:<8.3f}{brier:<8.3f}")