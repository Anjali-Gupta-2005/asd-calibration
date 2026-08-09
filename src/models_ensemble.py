"""
models_ensemble.py
Trains the ensemble models (XGBoost, AdaBoost, MLP) per cohort per fold,
using the pre-built train/calib/test splits from Phase 1. Saves raw
(uncalibrated) probabilities via utils.save_probs() for later calibration.

Imbalance handling (per the project's agreed decision - no SMOTE):
  - XGBoost: scale_pos_weight computed per training fold
  - AdaBoost: sklearn's AdaBoostClassifier has no class_weight param, so we
    give it a weak learner (depth-1 tree) with class_weight='balanced'
  - MLP: sklearn's MLPClassifier supports neither class_weight nor
    sample_weight - this is a known, documented limitation, left as-is
    rather than working around it with resampling (which would reintroduce
    the SMOTE-style distortion we explicitly decided against)

Run with: python src/models_ensemble.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import utils

# models sensitive to feature scale (age ranges ~5-90 while every other
# feature is 0/1 binary/one-hot) - gradient-based models need scaling,
# tree-based models (xgb, adaboost) are scale-invariant and don't
SCALE_SENSITIVE_MODELS = {"mlp"}


def load_cohort_data(cohort):
    df = pd.read_csv(os.path.join(cfg.PATH_PROCESSED, f"{cohort}_clean.csv"))
    y = df["class_asd"].astype(int).values
    X = df.drop(columns=["class_asd"]).astype(float).values
    return X, y


def load_fold_split(cohort, fold):
    path = os.path.join(cfg.PATH_SPLITS, f"{cohort}_fold{fold}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_model(name, y_train):
    if name == "xgb":
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.SEED,
            eval_metric="logloss",
        )
    elif name == "adaboost":
        weak_learner = DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=cfg.SEED
        )
        return AdaBoostClassifier(estimator=weak_learner, random_state=cfg.SEED)
    elif name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=cfg.SEED,
        )
    else:
        raise ValueError(f"Unknown ensemble model name: {name}")


def train_and_save(cohort, model_name, fold):
    X, y = load_cohort_data(cohort)
    splits = load_fold_split(cohort, fold)
    train_idx = splits["train_idx"]
    calib_idx = splits["calib_idx"]
    test_idx = splits["test_idx"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_calib, y_calib = X[calib_idx], y[calib_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if model_name in SCALE_SENSITIVE_MODELS:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)   # fit on train fold only
        X_calib = scaler.transform(X_calib)
        X_test = scaler.transform(X_test)

    model = build_model(model_name, y_train)
    model.fit(X_train, y_train)

    calib_probs = model.predict_proba(X_calib)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    utils.save_probs(cohort, model_name, fold, "calib", calib_probs, y_calib)
    utils.save_probs(cohort, model_name, fold, "test", test_probs, y_test)

    # sanity-check diagnostic only - NOT the project's actual reported metric
    # (that is calibration error, computed later in Phase 4). This just
    # catches a model that silently failed to learn anything (~50% accuracy
    # on a roughly balanced cohort would be a red flag worth investigating).
    test_preds = (test_probs >= 0.5).astype(int)
    accuracy = (test_preds == y_test).mean()
    return accuracy


def main():
    utils.set_seed()
    for cohort in cfg.COHORTS:
        print(f"\nCohort: {cohort}")
        for model_name in cfg.MODELS_ENSEMBLE:
            fold_accuracies = []
            for fold in range(cfg.N_FOLDS):
                acc = train_and_save(cohort, model_name, fold)
                fold_accuracies.append(acc)
            mean_acc = sum(fold_accuracies) / len(fold_accuracies)
            print(f"  {model_name}: done ({cfg.N_FOLDS} folds) "
                  f"- sanity-check test accuracy (uncalibrated, threshold 0.5): {mean_acc:.1%}")

    print("\nAll ensemble models trained and predictions saved to predictions/")
    print("Note: accuracy above is a sanity check only. The project's actual")
    print("reported metrics (ECE/MCE/Brier) come from Phase 4 evaluation.")


if __name__ == "__main__":
    main()