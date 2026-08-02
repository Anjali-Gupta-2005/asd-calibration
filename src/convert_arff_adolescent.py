"""
convert_arff_adolescent.py
One-off converter for the UCI Adolescent ASD dataset. This dataset is NOT
available through ucimlrepo's direct fetch (it has no registered data_url),
so it must be downloaded manually as a .arff file and converted here.

Manual step first:
  1. Go to https://archive.ics.uci.edu/dataset/420/autistic+spectrum+disorder+screening+data+for+adolescent
  2. Click Download, unzip it
  3. Find "Autism-Adolescent-Data.arff" and save it at exactly: data/raw/adolescent_raw.arff

Then run: python src/convert_arff_adolescent.py
"""

import os
import pandas as pd
from scipy.io import arff

RAW_ARFF_PATH = "data/raw/adolescent_raw.arff"
OUT_CSV_PATH = "data/raw/adolescent_raw.csv"


def main():
    if not os.path.exists(RAW_ARFF_PATH):
        raise FileNotFoundError(
            f"{RAW_ARFF_PATH} not found. Manually download "
            "Autism-Adolescent-Data.arff from "
            "https://archive.ics.uci.edu/dataset/420 and save it at this "
            "exact path first."
        )

    data, meta = arff.loadarff(RAW_ARFF_PATH)
    df = pd.DataFrame(data)

    # scipy loads string/categorical ARFF columns as byte-strings (b'yes')
    # instead of normal strings - decode every object column
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
            )

    # standardize the target column name so it matches adult/child raw files
    for candidate in ["class", "Class/ASD", "Class"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "Class/ASD"})
            break

    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"saved {OUT_CSV_PATH}  shape={df.shape}")
    print(f"columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()