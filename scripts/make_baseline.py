"""Generate src/api/baseline_stats.json for drift detection.

Creates mean/std in the *transformed model feature space*.

Matches inference:
  - load preprocessor.pkl
  - add engineered columns (house_age, bed_bath_ratio, price_per_sqft)
  - preprocessor.transform(...)
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

DATA_PATH = "data/raw/house_data.csv"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"
OUT_PATH = "src/api/baseline_stats.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    required = ["sqft", "bedrooms", "bathrooms", "location", "year_built", "condition"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns in {DATA_PATH}: {missing}\nColumns are: {df.columns.tolist()}"
        )

    # Match inference feature engineering
    df = df.copy()
    df["house_age"] = datetime.now().year - df["year_built"]
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    df["bed_bath_ratio"] = df["bed_bath_ratio"].fillna(0)
    df["price_per_sqft"] = 0  # dummy compatibility feature

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X = preprocessor.transform(df)

    # Convert sparse to dense
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-9, 1e-9, stds)

    features = [str(i) for i in range(X.shape[1])]
    baseline = {
        f: {"mean": float(means[i]), "std": float(stds[i])}
        for i, f in enumerate(features)
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump({"features": features, "baseline": baseline}, fp, indent=2)

    print(f"✅ wrote {OUT_PATH} with {X.shape[1]} features")


if __name__ == "__main__":
    main()
