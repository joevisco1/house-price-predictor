import argparse
import os
import platform
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# MLflow OPTIONAL
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


# -----------------------------
# Model factory
# -----------------------------
def get_model_instance(name: str, params: dict):
    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


# -----------------------------
# Preprocessor
# -----------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


# -----------------------------
# Feature engineering
# -----------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "year_built" in df.columns:
        df["house_age"] = datetime.now().year - df["year_built"]

    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        denom = df["bathrooms"].replace(0, 1)
        df["bed_bath_ratio"] = df["bedrooms"] / denom

    df["price_per_sqft"] = 0

    return df


# -----------------------------
# Training
# -----------------------------
def train_and_evaluate(model_cfg: dict, target: str, data_path: str):
    CONTRACT_FAIL_PREFIX = "TRAINING_CONTRACT_VIOLATION: "

    data = pd.read_csv(data_path)

    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset")

    data = apply_feature_engineering(data)

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # ---- SAVE X_train.npy (for baseline step) ----
    trained_dir = Path("models/trained")
    trained_dir.mkdir(parents=True, exist_ok=True)

    X_train_np = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
    X_train_np = np.asarray(X_train_np, dtype=float)
    np.save(trained_dir / "X_train.npy", X_train_np)
    logger.info("Saved X_train.npy")

    # ---- CONTRACT CHECK ----
    if X_train_t.shape[1] != X_test_t.shape[1]:
        raise RuntimeError(
            f"{CONTRACT_FAIL_PREFIX}transform width differs train={X_train_t.shape[1]} test={X_test_t.shape[1]}"
        )

    model = get_model_instance(model_cfg["best_model"], model_cfg["parameters"])
    logger.info(f"Training model: {model_cfg['best_model']}")

    model.fit(X_train_t, y_train)

    expected = int(X_train_t.shape[1])
    got = getattr(model, "n_features_in_", expected)

    if int(got) != expected:
        raise RuntimeError(
            f"{CONTRACT_FAIL_PREFIX}model expects {got} but preprocessor emits {expected}"
        )

    y_pred = model.predict(X_test_t)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return model, preprocessor, mae, r2


# -----------------------------
# Main
# -----------------------------
def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    target = model_cfg["target_variable"]

    model, preprocessor, mae, r2 = train_and_evaluate(model_cfg, target, args.data)

    trained_dir = os.path.join(args.models_dir, "trained")
    os.makedirs(trained_dir, exist_ok=True)

    model_path = os.path.join(trained_dir, "house_price_model.pkl")
    preprocessor_path = os.path.join(trained_dir, "preprocessor.pkl")
    bundle_path = os.path.join(trained_dir, "model_bundle.pkl")
    bundle_tmp = bundle_path + ".tmp"

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = None

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": {
            "expected_feature_count": int(getattr(model, "n_features_in_", -1)),
            "feature_names_out": feature_names,
            "input_columns": list(getattr(preprocessor, "feature_names_in_", [])),
        },
    }

    joblib.dump(bundle, bundle_tmp)
    os.replace(bundle_tmp, bundle_path)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Saved model bundle to: {bundle_path}")
    logger.info(f"Saved trained model to: {model_path}")
    logger.info(f"Saved preprocessor to: {preprocessor_path}")
    logger.info(f"Final MAE: {mae:.2f}, R²: {r2:.4f}")

    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed; skipping registry.")
        return


if __name__ == "__main__":
    args = parse_args()
    main(args)
