import argparse
import os
import platform
import logging
from datetime import datetime

import joblib

# MLflow is OPTIONAL (so training works inside the inference image)
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

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

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained artifacts")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()


# -----------------------------
# Load model from config
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
# Preprocessor (dtype-driven, no hard-coded feature list)
# -----------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    IMPORTANT:
    - remainder="drop" ensures no silent passthrough columns (the source of your 17 vs 16 mismatch).
    - numeric/categorical columns are derived from dtypes of the *training* frame after feature engineering.
    """
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
# Feature engineering (must match inference)
# -----------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "year_built" in df.columns:
        df["house_age"] = datetime.now().year - df["year_built"]

    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        denom = df["bathrooms"].replace(0, 1)
        df["bed_bath_ratio"] = df["bedrooms"] / denom

    # Keep constant to match inference behavior and avoid leakage
    df["price_per_sqft"] = 0

    return df


# -----------------------------
# Core training routine (works with or without MLflow)
# -----------------------------
def train_and_evaluate(model_cfg: dict, target: str, data_path: str):
    CONTRACT_FAIL_PREFIX = "TRAINING_CONTRACT_VIOLATION: "

    data = pd.read_csv(data_path)

    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset columns: {list(data.columns)}")

    # Must match inference
    data = apply_feature_engineering(data)

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # --- CONTRACT CHECKS (must never ship mismatched artifacts) ---
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

    # Train + eval (no mlflow dependency here)
    model, preprocessor, mae, r2 = train_and_evaluate(model_cfg, target, args.data)

    # Save artifacts locally for container deployment (canonical filenames)
    trained_dir = os.path.join(args.models_dir, "trained")
    os.makedirs(trained_dir, exist_ok=True)

    model_path = os.path.join(trained_dir, "house_price_model.pkl")
    preprocessor_path = os.path.join(trained_dir, "preprocessor.pkl")

    # --- ATOMIC BUNDLE (source of truth) ---
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

    # canonical artifacts (same in-memory objects, so contract cannot drift)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Saved model bundle to: {bundle_path}")
    logger.info(f"Saved trained model to: {model_path}")
    logger.info(f"Saved preprocessor to: {preprocessor_path}")
    logger.info(f"Final MAE: {mae:.2f}, R²: {r2:.4f}")

    # Optional MLflow logging/registry (only if installed)
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed in this environment; skipping MLflow logging/registry.")
        return

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(model_cfg["name"])

    with mlflow.start_run(run_name="final_training"):
        mlflow.log_params(model_cfg["parameters"])
        mlflow.log_metrics({"mae": mae, "r2": r2})

        # Log estimator artifact (keeps registry informative)
        mlflow.sklearn.log_model(model, "tuned_model")

        model_name = model_cfg["name"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass  # already exists or registry not available

        try:
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id,
            )

            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
            )
        except Exception as e:
            logger.warning(f"MLflow registry operations failed (continuing): {e}")

        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All dataset features except target + engineered fields (house_age, bed_bath_ratio, price_per_sqft)\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Saved artifacts:\n"
            f"  - Bundle: {bundle_path}\n"
            f"  - Model: {model_path}\n"
            f"  - Preprocessor: {preprocessor_path}\n"
            f"Performance metrics:\n"
            f"  - MAE: {mae:.2f}\n"
            f"  - R²: {r2:.4f}"
        )

        try:
            client.update_registered_model(name=model_name, description=description)
            client.set_registered_model_tag(model_name, "algorithm", model_cfg["best_model"])
            client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg["parameters"]))
            client.set_registered_model_tag(
                model_name,
                "features",
                "all_except_target_plus_engineered(house_age, bed_bath_ratio, price_per_sqft)",
            )
            client.set_registered_model_tag(model_name, "target_variable", target)
            client.set_registered_model_tag(model_name, "training_dataset", args.data)
            client.set_registered_model_tag(model_name, "bundle_path", bundle_path)
            client.set_registered_model_tag(model_name, "model_path", model_path)
            client.set_registered_model_tag(model_name, "preprocessor_path", preprocessor_path)

            deps = {
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "xgboost_version": xgb.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            }
            for k, v in deps.items():
                client.set_registered_model_tag(model_name, k, v)
        except Exception as e:
            logger.warning(f"MLflow model metadata/tagging failed (continuing): {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
