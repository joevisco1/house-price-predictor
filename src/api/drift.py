"""
Simple online drift metric exported to Prometheus.

We compute a single scalar drift score as the average absolute z-score across the
TRANSFORMED feature vector (the exact feature space your model sees).

If baseline stats are missing, we DO NOT crash the API; we expose a metric flag
`model_drift_baseline_loaded` so you can alert on misconfiguration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal

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

# Synthetic traffic execution counter
SYNTH_REQUESTS_TOTAL = Counter(
    "model_synthetic_requests_total",
    "Synthetic traffic requests executed (normal/drift) against active/preview, labeled by outcome.",
    ["service", "kind", "status"],
)

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "3.0"))
DRIFT_HIGH_THRESHOLD.set(DRIFT_THRESHOLD)
BASELINE_PATH = os.getenv("BASELINE_PATH", "baseline_stats.json")


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


def record_synth_request(
    service: str,
    kind: Literal["normal", "drift"],
    status: Literal["success", "failure"],
) -> None:
    """Increment synthetic-traffic counter with bounded labels."""
    svc = service.strip()[:64] if service else "unknown"
    k = kind if kind in ("normal", "drift") else "normal"
    s = status if status in ("success", "failure") else "failure"
    SYNTH_REQUESTS_TOTAL.labels(service=svc, kind=k, status=s).inc()


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

    zscores = []
    for i, f in enumerate(features):
        try:
            x = float(row[i])
            mu = float(stats[f]["mean"])
            sigma = float(stats[f]["std"]) or 1e-9
            zscores.append(abs((x - mu) / sigma))
        except Exception:
            continue

        # Make score responsive: mean of top-K z-scores (dramatic, still stable)
    if not zscores:
        score = 0.0
    else:
        zscores.sort(reverse=True)
        k = min(25, len(zscores))  # top 25 features
        score = sum(zscores[:k]) / float(k)
    DRIFT_SCORE.set(score)

    if score >= DRIFT_THRESHOLD:
        DRIFT_HIGH_TOTAL.inc()

    return score


def get_drift_snapshot() -> Dict[str, float]:
    """
    Small JSON-friendly snapshot for UI/debug.

    Note: prometheus_client does not provide a public read API for metric values,
    so we read the underlying value holders.
    """
    try:
        score = float(DRIFT_SCORE._value.get())  # type: ignore[attr-defined]
    except Exception:
        score = 0.0

    try:
        threshold = float(DRIFT_HIGH_THRESHOLD._value.get())  # type: ignore[attr-defined]
    except Exception:
        threshold = float(DRIFT_THRESHOLD)

    try:
        baseline_loaded = float(BASELINE_LOADED._value.get())  # type: ignore[attr-defined]
    except Exception:
        baseline_loaded = 0.0

    try:
        high_total = float(DRIFT_HIGH_TOTAL._value.get())  # type: ignore[attr-defined]
    except Exception:
        high_total = 0.0

    return {
        "drift_score": score,
        "drift_threshold": threshold,
        "drift_high_total": high_total,
        "baseline_loaded": baseline_loaded,
    }
