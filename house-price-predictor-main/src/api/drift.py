"""Simple online drift metric exported to Prometheus.

We compute a single scalar drift score as the average absolute z-score across the
TRANSFORMED feature vector (the exact feature space your model sees).

If baseline stats are missing or incompatible, we DO NOT crash the API; we expose
`model_drift_baseline_loaded` so you can alert on misconfiguration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

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

BASELINE_LOADED = Gauge(
    "model_drift_baseline_loaded",
    "1 if baseline_stats.json loaded successfully and matches features, else 0.",
)

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "3.0"))


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


def _find_baseline() -> Dict[str, Any] | None:
    # Prefer explicit env var
    env_path = os.getenv("BASELINE_PATH")
    if env_path:
        b = _safe_load_baseline(env_path)
        if b is not None:
            return b

    # Common locations for repo runs and container runs
    candidates = [
        "baseline_stats.json",
        "src/api/baseline_stats.json",
        "models/trained/baseline_stats.json",
        "/app/baseline_stats.json",
        "/app/src/api/baseline_stats.json",
        "/app/models/trained/baseline_stats.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            b = _safe_load_baseline(p)
            if b is not None:
                return b
    return None


_BASELINE: Dict[str, Any] | None = _find_baseline()
BASELINE_LOADED.set(1 if _BASELINE is not None else 0)


def record_drift_metrics(processed_features) -> float:
    """Record drift metrics from the output of `preprocessor.transform(...)`.

    `processed_features` may be a sparse matrix or dense array. We use the first row.
    """
    if _BASELINE is None:
        DRIFT_SCORE.set(0.0)
        return 0.0

    features = _BASELINE.get("features")
    stats = _BASELINE.get("baseline")
    if not isinstance(features, list) or not isinstance(stats, dict):
        BASELINE_LOADED.set(0)
        DRIFT_SCORE.set(0.0)
        return 0.0

    # Get first row
    row = processed_features[0]
    if hasattr(row, "toarray"):
        row = row.toarray()[0]
    if hasattr(row, "ravel"):
        row = row.ravel()

    # Guard: baseline feature count must match transformed vector length
    try:
        if len(features) > len(row):
            BASELINE_LOADED.set(0)
            DRIFT_SCORE.set(0.0)
            return 0.0
    except Exception:
        BASELINE_LOADED.set(0)
        DRIFT_SCORE.set(0.0)
        return 0.0

    zscores = []
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
