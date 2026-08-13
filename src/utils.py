"""
utils.py
Shared helper functions used by every phase (model training, calibration,
evaluation). This is the "contract" both of you write to/read from, so
neither person needs to know how the other's code works internally.

Import with: from src import utils   (or "import utils" if run from inside src/)
"""

import os
import random
import numpy as np

import config as cfg
import pandas as pd
import pickle
from config import PATH_PROCESSED, PATH_SPLITS

def set_seed(seed=cfg.SEED):
    """Sets every relevant random seed. Call this at the top of every script."""
    random.seed(seed)
    np.random.seed(seed)


def _check_valid_name(cohort, model):
    if cohort not in cfg.COHORTS:
        raise ValueError(f"'{cohort}' is not a recognized cohort. Use one of {cfg.COHORTS}")
    if model not in cfg.ALL_MODELS:
        raise ValueError(f"'{model}' is not a recognized model. Use one of {cfg.ALL_MODELS}")


def save_probs(cohort, model, fold, slice_name, probs, labels):
    """Saves raw (uncalibrated) model probabilities + true labels.
    slice_name must be 'calib' or 'test'."""
    _check_valid_name(cohort, model)
    if slice_name not in ("calib", "test"):
        raise ValueError("slice_name must be 'calib' or 'test'")

    os.makedirs(cfg.PATH_PREDICTIONS, exist_ok=True)
    base = f"{cohort}_{model}_fold{fold}_{slice_name}slice"
    np.save(os.path.join(cfg.PATH_PREDICTIONS, f"{base}_probs.npy"), np.asarray(probs))
    np.save(os.path.join(cfg.PATH_PREDICTIONS, f"{base}_labels.npy"), np.asarray(labels))


def load_probs(cohort, model, fold, slice_name):
    """Returns (probs, labels) - inverse of save_probs."""
    _check_valid_name(cohort, model)
    base = f"{cohort}_{model}_fold{fold}_{slice_name}slice"
    probs = np.load(os.path.join(cfg.PATH_PREDICTIONS, f"{base}_probs.npy"))
    labels = np.load(os.path.join(cfg.PATH_PREDICTIONS, f"{base}_labels.npy"))
    return probs, labels


def save_calibrated_probs(cohort, model, method, fold, probs):
    """Saves calibrated test-slice probabilities for one (cohort, model, method, fold)."""
    _check_valid_name(cohort, model)
    if method not in cfg.ALL_METHODS:
        raise ValueError(f"'{method}' is not a recognized calibration method. Use one of {cfg.ALL_METHODS}")

    os.makedirs(cfg.PATH_CALIBRATED, exist_ok=True)
    fname = f"{cohort}_{model}_{method}_fold{fold}_testslice_probs.npy"
    np.save(os.path.join(cfg.PATH_CALIBRATED, fname), np.asarray(probs))


def load_calibrated_probs(cohort, model, method, fold):
    """Returns the calibrated probs array - inverse of save_calibrated_probs."""
    _check_valid_name(cohort, model)
    fname = f"{cohort}_{model}_{method}_fold{fold}_testslice_probs.npy"
    return np.load(os.path.join(cfg.PATH_CALIBRATED, fname))

def load_fold(cohort, fold):
    """Loads clean data + split indices for one cohort/fold. Returns (df, idx_dict)."""
    df = pd.read_csv(f"{PATH_PROCESSED}{cohort}_clean.csv")
    with open(f"{PATH_SPLITS}{cohort}_fold{fold}.pkl", "rb") as f:
        idx = pickle.load(f)
    return df, idx