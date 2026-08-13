import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from config import COHORTS,  MODELS_CLASSICAL, N_FOLDS, SEED
from src.utils import set_seed, load_fold, save_probs
from sklearn.preprocessing import StandardScaler

TARGET_COL = "class_asd"

MODEL_BUILDERS = {
    "logreg": lambda: LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000),
    "dtree":  lambda: DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    "knn":    lambda: KNeighborsClassifier(),
    "nb":     lambda: GaussianNB(),
    "rf":     lambda: RandomForestClassifier(class_weight="balanced", random_state=SEED, n_estimators=200),
    "svm":    lambda: SVC(probability=True, class_weight="balanced", random_state=SEED),
}

def train_model(cohort, model_name):
    set_seed()
    for fold in range(N_FOLDS):
        df, idx = load_fold(cohort, fold)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL].values

        X_train, y_train = X.iloc[idx["train_idx"]], y[idx["train_idx"]]
        X_calib, y_calib = X.iloc[idx["calib_idx"]], y[idx["calib_idx"]]
        X_test,  y_test  = X.iloc[idx["test_idx"]],  y[idx["test_idx"]]

        if model_name in {"knn", "svm"}:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_calib = scaler.transform(X_calib)
            X_test = scaler.transform(X_test)

        clf = MODEL_BUILDERS[model_name]()

        if model_name == "nb":
            sw = compute_sample_weight("balanced", y_train)
            clf.fit(X_train, y_train, sample_weight=sw)
        else:
            clf.fit(X_train, y_train)

        calib_probs = clf.predict_proba(X_calib)[:, 1]
        test_probs  = clf.predict_proba(X_test)[:, 1]

        save_probs(cohort, model_name, fold, "calib", calib_probs, y_calib)
        save_probs(cohort, model_name, fold, "test", test_probs, y_test)

if __name__ == "__main__":
    for cohort, model in itertools.product(COHORTS, MODELS_CLASSICAL):
        print(f"Training {model} on {cohort}...")
        train_model(cohort, model)
    print("Done. 6 models x 4 cohorts x 3 folds = 72 file-pairs.")