"""
Inference + drift metrics exported to Prometheus.

- Exposes:
  - predict_price(payload: dict) -> dict
  - batch_predict(payloads: list[dict]) -> list[dict]

- Drift:
  Avg abs z-score across the TRANSFORMED feature vector (the model's feature space).

- Robustness:
  If baseline stats are missing, we DO NOT crash; we export `model_drift_baseline_loaded`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
from prometheus_client import Counter, Gauge

# --- Prometheus metrics ---
DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Avg abs z-score across transformed model features vs training baseline.",
)

DRIFT_HIGH_TOTAL = Counter(
    "model_drift_high_total",
    "Count of requests where drift score exceeded threshold.",
)

DRIFT_HIGH_THRESHOLD = Gauge(
    "model_drift_high_threshold",
    "Threshold used to increment model_drift_high_total.",
)

BASELINE_LOADED = Gauge(
    "model_drift_baseline_loaded",
    "1 if baseline_stats.json loaded successfully, else 0.",
)

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "3.0"))
BASELINE_PATH = os.getenv("BASELINE_PATH", "baseline_stats.json")

# Model artifacts (copied into image at /app/models/trained/*)
MODEL_DIR = os.getenv("MODEL_DIR", "models/trained")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", os.path.join(MODEL_DIR, "preprocessor.pkl"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))


def _safe_load_baseline(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            baseline = json.load(fp)
        if "features" not in baseline or "baseline" not in baseline:
            return None
        return baseline
    except FileNotFoundError:
        return None
    except Exception:
        return None


_BASELINE: Dict[str, Any] | None = _safe_load_baseline(BASELINE_PATH)
BASELINE_LOADED.set(1 if _BASELINE is not None else 0)


def record_drift_metrics(processed_features) -> float:
    """
    Record drift metrics from the output of `preprocessor.transform(...)`.

    `processed_features` may be a sparse matrix or dense array. We use the first row.
    """
    DRIFT_HIGH_THRESHOLD.set(DRIFT_THRESHOLD)

    if _BASELINE is None:
        DRIFT_SCORE.set(0.0)
        return 0.0

    features = _BASELINE["features"]  # e.g., ["0","1",...]
    stats = _BASELINE["baseline"]

    row = processed_features[0]

    # scipy sparse row => dense
    if hasattr(row, "toarray"):
        row = row.toarray()[0]

    if hasattr(row, "ravel"):
        row = row.ravel()

    zscores: List[float] = []
    for i, f in enumerate(features):
        try:
            x = float(row[i])
            mu = float(stats[f]["mean"])
            sigma = float(stats[f]["std"]) or 1e-9
            zscores.append(abs((x - mu) / sigma))
        except Exception:
            continue

    score = sum(zscores) / len(zscores) if zscores else 0.0
    DRIFT_SCORE.set(score)

    if score >= DRIFT_THRESHOLD:
        DRIFT_HIGH_TOTAL.inc()

    return score


# --- Lazy-loaded model + preprocessor (cached) ---
_PREPROCESSOR = None
_MODEL = None


def _load_artifacts() -> Tuple[Any, Any]:
    """Load preprocessor + model once per process."""
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
    """Convert API payload to model feature dict."""
    return {
        "sqft": payload.get("sqft"),
        "bedrooms": payload.get("bedrooms"),
        "bathrooms": payload.get("bathrooms"),
        "location": payload.get("location"),
        "year_built": payload.get("year_built"),
        "condition": payload.get("condition"),
    }


def _predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Core inference for a single payload."""
    preprocessor, model = _load_artifacts()
    features = _payload_to_features(payload)

    try:
        X = preprocessor.transform([features])
    except Exception as e:
        raise RuntimeError(f"Preprocess failed: {e}")

    drift_score = record_drift_metrics(X)

    try:
        y = model.predict(X)
        pred = float(y[0])
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")

    # Return fields that satisfy common schemas.
    # Keep 'prediction' (your current main.py extractor supports it)
    # Add 'price' as a harmless alias in case your schema expects it.
    return {
        "prediction": pred,
        "price": pred,
        "drift_score": drift_score,
    }


# --- Exports expected by /app/main.py ---
def predict_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _predict_one(payload)


def batch_predict(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_predict_one(p) for p in payloads]
