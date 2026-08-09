"""
check_predictions.py
Run this after models_classical.py / models_ensemble.py, before committing.
Verifies every (cohort, model, fold) combination produced valid, correctly
sized prediction files.

Run with: python src/check_predictions.py
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import utils

ISSUES = []


def check(condition, message):
    if condition:
        print(f"  OK   - {message}")
    else:
        print(f"  FAIL - {message}")
        ISSUES.append(message)


def get_expected_split_sizes(cohort, fold):
    path = os.path.join(cfg.PATH_SPLITS, f"{cohort}_fold{fold}.pkl")
    with open(path, "rb") as f:
        splits = pickle.load(f)
    return len(splits["calib_idx"]), len(splits["test_idx"])


def main():
    for cohort in cfg.COHORTS:
        print(f"\nCohort: {cohort}")
        for model in cfg.ALL_MODELS:
            expected_calib_n, expected_test_n = None, None
            model_files_exist = True

            for fold in range(cfg.N_FOLDS):
                expected_calib_n, expected_test_n = get_expected_split_sizes(cohort, fold)

                for slice_name, expected_n in [("calib", expected_calib_n), ("test", expected_test_n)]:
                    try:
                        probs, labels = utils.load_probs(cohort, model, fold, slice_name)
                    except FileNotFoundError:
                        model_files_exist = False
                        continue

                    check(
                        len(probs) == expected_n,
                        f"{cohort}/{model}/fold{fold}/{slice_name}: length {len(probs)} matches expected {expected_n}",
                    )
                    check(
                        (probs >= 0).all() and (probs <= 1).all(),
                        f"{cohort}/{model}/fold{fold}/{slice_name}: all probabilities in [0,1]",
                    )
                    check(
                        set(np.unique(labels)) <= {0, 1},
                        f"{cohort}/{model}/fold{fold}/{slice_name}: labels strictly binary",
                    )

            if not model_files_exist:
                print(f"  (skipped {model}: no prediction files found yet)")

    print("\n" + "=" * 50)
    if ISSUES:
        print(f"{len(ISSUES)} ISSUE(S) FOUND:")
        for issue in ISSUES:
            print(f"  - {issue}")
    else:
        print("All prediction files valid. Safe to commit.")


if __name__ == "__main__":
    main()