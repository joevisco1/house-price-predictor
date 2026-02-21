"""
Inference functions used by FastAPI entrypoint.

Exports:
  - predict_price(payload: dict) -> dict
  - batch_predict(payloads: list[dict]) -> list[dict]

Drift metrics are defined ONLY in drift.py.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

from drift import record_drift_metrics  # single source of truth for drift metrics

MODEL_DIR = os.getenv("MODEL_DIR", "models/trained")

# Primary artifacts
PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH",
    os.path.join(MODEL_DIR, "preprocessor.pkl"),
)

# MODEL_PATH may be overridden by env; if not, default to model_bundle.pkl (your newer artifact)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(MODEL_DIR, "model_bundle.pkl"),
)

# Fallbacks for older training outputs / naming
_MODEL_FALLBACKS = [
    os.path.join(MODEL_DIR, "model_bundle.pkl"),
    os.path.join(MODEL_DIR, "house_price_model.pkl"),
    os.path.join(MODEL_DIR, "model.pkl"),
]

_PREPROCESSOR: Any = None
_MODEL: Any = None


def _resolve_model_path() -> str:
    # 1) If env/default MODEL_PATH exists, use it.
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        return MODEL_PATH

    # 2) Try known fallback filenames.
    for p in _MODEL_FALLBACKS:
        if os.path.exists(p):
            return p

    # 3) Nothing found.
    tried = [MODEL_PATH] + _MODEL_FALLBACKS
    tried = [t for t in tried if t]
    raise RuntimeError(
        "Missing model artifact. Tried:\n  - " + "\n  - ".join(tried)
    )


def _load_artifacts() -> Tuple[Any, Any]:
    global _PREPROCESSOR, _MODEL

    # Load model first because model_bundle.pkl includes BOTH model + preprocessor.
    if _MODEL is None:
        resolved_model_path = _resolve_model_path()
        obj = joblib.load(resolved_model_path)

        # Newer training output: bundle dict with keys: metadata, model, preprocessor
        if isinstance(obj, dict) and "model" in obj and "preprocessor" in obj:
            _MODEL = obj["model"]
            if _PREPROCESSOR is None:
                _PREPROCESSOR = obj["preprocessor"]
        else:
            _MODEL = obj

    # If preprocessor not provided by bundle, load it from its own artifact.
    if _PREPROCESSOR is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise RuntimeError(f"Missing preprocessor artifact: {PREPROCESSOR_PATH}")
        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

    return _PREPROCESSOR, _MODEL


def _payload_to_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sqft": payload.get("sqft"),
        "bedrooms": payload.get("bedrooms"),
        "bathrooms": payload.get("bathrooms"),
        "location": payload.get("location"),
        "year_built": payload.get("year_built"),
        "condition": payload.get("condition"),
    }


def _predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    preprocessor, model = _load_artifacts()
    features = _payload_to_features(payload)

    # Preprocessor expects 2D tabular input with engineered columns
    df = pd.DataFrame([features])

    current_year = datetime.utcnow().year

    # engineered columns required by the trained preprocessor
    if "year_built" in df.columns and df["year_built"].notna().all():
        df["house_age"] = current_year - df["year_built"]
    else:
        df["house_age"] = 0

    # unknown at inference time; keep neutral placeholder
    df["price_per_sqft"] = 0.0

    # avoid division by zero
    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        denom = df["bathrooms"].replace(0, 1)
        df["bed_bath_ratio"] = df["bedrooms"] / denom
    else:
        df["bed_bath_ratio"] = 0.0

    X = preprocessor.transform(df)

    # drift computed + exported here (metrics live in drift.py)
    drift_score = record_drift_metrics(X)

    y = model.predict(X)
    pred = float(y[0])

    return {"prediction": pred, "price": pred, "drift_score": drift_score}


def predict_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _predict_one(payload)


def batch_predict(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_predict_one(p) for p in payloads]
