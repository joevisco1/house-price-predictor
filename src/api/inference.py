"""
Inference functions used by FastAPI entrypoint.

Exports:
  - predict_price(payload: dict) -> dict
  - batch_predict(payloads: list[dict]) -> list[dict]

Drift metrics are defined ONLY in drift.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import joblib

from drift import record_drift_metrics  # <-- single source of truth for drift metrics

MODEL_DIR = os.getenv("MODEL_DIR", "models/trained")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", os.path.join(MODEL_DIR, "preprocessor.pkl"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))

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

    X = preprocessor.transform([features])

    # drift computed + exported here (metrics live in drift.py)
    drift_score = record_drift_metrics(X)

    y = model.predict(X)
    pred = float(y[0])

    return {"prediction": pred, "price": pred, "drift_score": drift_score}


def predict_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _predict_one(payload)


def batch_predict(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_predict_one(p) for p in payloads]
