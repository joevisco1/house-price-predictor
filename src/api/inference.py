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

from drift import record_drift_metrics  # <-- single source of truth for drift metrics

MODEL_DIR = os.getenv("MODEL_DIR", "models/trained")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", os.path.join(MODEL_DIR, "preprocessor.pkl"))

# FIX 1: default model artifact name matches the container artifact
# (still allows override via MODEL_PATH env var)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "model_bundle.pkl"))

_PREPROCESSOR = None
_MODEL = None


def _load_artifacts() -> Tuple[Any, Any]:
    global _PREPROCESSOR, _MODEL

    if _PREPROCESSOR is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise RuntimeError(f"Missing preprocessor artifact: {PREPROCESSOR_PATH}")
        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Missing model artifact: {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)

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

    # FIX 2: preprocessor expects a 2D tabular input with engineered columns
    df = pd.DataFrame([features])

    current_year = datetime.utcnow().year

    # engineered columns required by the trained preprocessor
    if "year_built" in df.columns:
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
