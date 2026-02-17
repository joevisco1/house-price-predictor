import argparse
import os
import platform
import logging
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient
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
        # Avoid divide-by-zero and inf
        denom = df["bathrooms"].replace(0, 1)
        df["bed_bath_ratio"] = df["bedrooms"] / denom

    # Keep constant to match inference behavior and avoid label leakage
    df["price_per_sqft"] = 0

    return df


# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    target = model_cfg["target_variable"]

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Experiment name uses config model name
    mlflow.set_experiment(model_cfg["name"])

    # Load data
    data = pd.read_csv(args.data)

    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' not found in dataset columns: {list(data.columns)}")

    # Apply feature engineering BEFORE split, and BEFORE fitting preprocessor (must match inference)
    data = apply_feature_engineering(data)

    # Split features/target
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build preprocessor on training data (no schema guessing)
    preprocessor = build_preprocessor(X_train)

    # Transform
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Build model from config
    model = get_model_instance(model_cfg["best_model"], model_cfg["parameters"])

    # Ensure output directory exists
    trained_dir = os.path.join(args.models_dir, "trained")
    os.makedirs(trained_dir, exist_ok=True)

    # Canonical artifact paths expected by inference
    model_path = os.path.join(trained_dir, "house_price_model.pkl")
    preprocessor_path = os.path.join(trained_dir, "preprocessor.pkl")

    # Start MLflow run
    with mlflow.start_run(run_name="final_training"):
        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train_t, y_train)

        y_pred = model.predict(X_test_t)

        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        # Log params and metrics
        mlflow.log_params(model_cfg["parameters"])
        mlflow.log_metrics({"mae": mae, "r2": r2})

        # Log sklearn model artifact (estimator only). Registry remains informative.
        mlflow.sklearn.log_model(model, "tuned_model")

        model_name = model_cfg["name"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id,
        )

        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        # Human-readable description
        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All dataset features except target + engineered fields (house_age, bed_bath_ratio, price_per_sqft)\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Saved artifacts:\n"
            f"  - Model: {model_path}\n"
            f"  - Preprocessor: {preprocessor_path}\n"
            f"Performance metrics:\n"
            f"  - MAE: {mae:.2f}\n"
            f"  - R²: {r2:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        # Tags for organization
        client.set_registered_model_tag(model_name, "algorithm", model_cfg["best_model"])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg["parameters"]))
        client.set_registered_model_tag(
            model_name,
            "features",
            "all_except_target_plus_engineered(house_age, bed_bath_ratio, price_per_sqft)",
        )
        client.set_registered_model_tag(model_name, "target_variable", target)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)
        client.set_registered_model_tag(model_name, "model_path", model_path)
        client.set_registered_model_tag(model_name, "preprocessor_path", preprocessor_path)

        # Dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # Save artifacts locally for container deployment
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        logger.info(f"Saved trained model to: {model_path}")
        logger.info(f"Saved preprocessor to: {preprocessor_path}")
        logger.info(f"Final MAE: {mae:.2f}, R²: {r2:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
