import os
from datetime import datetime

import joblib
import pandas as pd

from schemas import HousePredictionRequest, PredictionResponse
from drift import record_drift_metrics

# ---- Model artifact locations ----
# Support both local repo runs (relative paths) and container runs (/app/...)
DEFAULT_MODEL_PATHS = [
    os.getenv("MODEL_PATH"),
    "models/trained/house_price_model.pkl",
    "/app/models/trained/house_price_model.pkl",
]
DEFAULT_PREPROCESSOR_PATHS = [
    os.getenv("PREPROCESSOR_PATH"),
    "models/trained/preprocessor.pkl",
    "/app/models/trained/preprocessor.pkl",
]


def _first_existing(paths: list[str | None]) -> str:
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            return p
    # fall back to first non-empty (helps error message)
    for p in paths:
        if p:
            return p
    raise RuntimeError("No candidate paths provided")


MODEL_PATH = _first_existing(DEFAULT_MODEL_PATHS)
PREPROCESSOR_PATH = _first_existing(DEFAULT_PREPROCESSOR_PATHS)

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {e}")


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same lightweight feature engineering as training."""
    out = df.copy()

    # engineered features (must match training expectations)
    out["house_age"] = datetime.now().year - out["year_built"].astype(int)

    # avoid any weird numeric issues (bathrooms is gt0 in schema, but keep safe anyway)
    bathrooms = out["bathrooms"].astype(float).replace(0, 1e-9)
    out["bed_bath_ratio"] = out["bedrooms"].astype(float) / bathrooms

    # compatibility column used in some pipelines
    if "price_per_sqft" not in out.columns:
        out["price_per_sqft"] = 0.0

    return out


def _align_to_preprocessor_input(df: pd.DataFrame) -> pd.DataFrame:
    """Align dataframe columns to what the fitted preprocessor expects.

    This prevents the 'number of features' crash that happens when:
      - training was fit on a DataFrame with specific columns, and
      - inference sends missing/extra columns.

    If the preprocessor exposes `feature_names_in_`, we use it.
    Otherwise we return df as-is and let transform raise a clear error.
    """
    expected = getattr(preprocessor, "feature_names_in_", None)
    if expected is None:
        return df

    expected = list(expected)

    # add missing columns (default 0)
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    # drop unexpected columns
    extra = [c for c in df.columns if c not in expected]
    if extra:
        df = df.drop(columns=extra)

    # order columns exactly
    return df[expected]


def _transform(df: pd.DataFrame):
    """Transform with best-effort alignment and helpful errors."""
    df = _engineer_features(df)
    df = _align_to_preprocessor_input(df)

    try:
        Xt = preprocessor.transform(df)
    except Exception as e:
        # Common failure: feature name mismatch when feature_names_in_ isn't present.
        # Re-raise with more context.
        raise RuntimeError(
            "Preprocessor transform failed. This usually means the inference input schema "
            "doesn't match what the preprocessor was trained on. "
            f"Model path={MODEL_PATH}, preprocessor path={PREPROCESSOR_PATH}. "
            f"Columns sent={list(df.columns)}. Original error: {e}"
        )

    return Xt


def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """Predict house price based on input features."""
    input_df = pd.DataFrame([request.dict()])

    processed_features = _transform(input_df)

    # record drift based on transformed feature space
    try:
        record_drift_metrics(processed_features)
    except Exception:
        # drift must never break inference
        pass

    predicted_price = model.predict(processed_features)[0]
    predicted_price = round(float(predicted_price), 2)

    confidence_interval = [round(predicted_price * 0.9, 2), round(predicted_price * 1.1, 2)]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat(),
    )


def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """Perform batch predictions."""
    input_df = pd.DataFrame([req.dict() for req in requests])
    processed_features = _transform(input_df)

    preds = model.predict(processed_features)
    return [float(p) for p in preds]
