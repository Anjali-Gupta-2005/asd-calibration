SEED = 42
N_FOLDS = 3
TRAIN_FRAC, CALIB_FRAC, TEST_FRAC = 0.6, 0.2, 0.2

COHORTS = ["toddler", "child", "adolescent", "adult"]

MODELS_CLASSICAL = ["logreg", "dtree", "knn", "nb", "rf", "svm"]
MODELS_ENSEMBLE = ["xgb", "adaboost", "mlp"]
ALL_MODELS = MODELS_CLASSICAL + MODELS_ENSEMBLE

METHODS_SIMPLE = ["platt", "temperature", "histbin"]
METHODS_FLEXIBLE = ["isotonic", "beta"]
ALL_METHODS = METHODS_SIMPLE + METHODS_FLEXIBLE

PATH_RAW = "data/raw/"
PATH_PROCESSED = "data/processed/"
PATH_SPLITS = "split_indices/"
PATH_PREDICTIONS = "predictions/"
PATH_CALIBRATED = "calibrated_predictions/"
PATH_RESULTS = "results/"

ECE_N_BINS = 10

# Final harmonized column set every processed CSV must have
# (country_of_res is absent for the toddler cohort only - that dataset never collected it)
CANONICAL_ITEMS = [f"A{i}" for i in range(1, 11)]
CANONICAL_COMMON = ["age", "sex", "ethnicity", "jaundice", "family_mem_with_ASD", "class_asd"]
