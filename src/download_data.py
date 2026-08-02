"""
download_data.py
Fetches the 3 UCI Autism Spectrum Disorder screening datasets directly via
the ucimlrepo package - no manual .arff conversion needed.

IMPORTANT: these are NOT the same as UCI id=2 ("Adult" / Census Income,
an unrelated income-prediction dataset that happens to share the name
"Adult"). The correct ids for the autism screening datasets are below.

The Kaggle Toddler dataset has no equivalent no-auth fetch method, so it
still needs a manual download - see the project README.

Run with: python src/download_data.py
"""

import os
from ucimlrepo import fetch_ucirepo

UCI_IDS = {
    "adult": 426,       # Autism Screening Adult
    "child": 419,       # Autistic Spectrum Disorder Screening Data for Children
    "adolescent": 420,  # Autistic Spectrum Disorder Screening Data for Adolescent
}

os.makedirs("data/raw", exist_ok=True)

for cohort, uci_id in UCI_IDS.items():
    print(f"Fetching {cohort} (UCI id={uci_id}) ...")
    dataset = fetch_ucirepo(id=uci_id)

    X = dataset.data.features.copy()
    y = dataset.data.targets

    # attach the label back on under a fixed, known name so preprocessing.py
    # doesn't need to guess what ucimlrepo called the target column
    X["Class/ASD"] = y.iloc[:, 0]

    out_path = f"data/raw/{cohort}_raw.csv"
    X.to_csv(out_path, index=False)
    print(f"  saved {out_path}  shape={X.shape}")
    print(f"  columns: {X.columns.tolist()}")
    print()

print("Done with UCI datasets (adult, child, adolescent).")
print("Toddler dataset still needs a manual Kaggle download - see README.")
