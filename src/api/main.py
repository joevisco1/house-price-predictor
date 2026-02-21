from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from typing import Any

from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from drift import record_synth_request

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features. "
        "Includes endpoints for single and batch predictions, as well as a health check. "
    ),
    version="1.0.0",
    contact={"name": "Joseph Visco"},
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exposes /metrics on the SAME app port (this is what your ServiceMonitor scrapes)
Instrumentator().instrument(app).expose(app)

# -------------------------
# Metrics
# -------------------------
model_requests_total = Counter("model_requests_total", "Total prediction requests")

model_feature_numeric = Histogram(
    "model_feature_numeric",
    "Numeric feature values",
    ["feature"],
    buckets=(0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000),
)

model_feature_categorical_total = Counter(
    "model_feature_categorical_total",
    "Categorical feature counts",
    ["feature", "value"],
)

model_prediction_price = Histogram(
    "model_prediction_price",
    "Predicted price distribution",
    buckets=(0, 100_000, 200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000, 1_250_000, 1_500_000, 2_000_000, 5_000_000),
)

# Set SERVICE_NAME in the container env to "model-active" or "model-preview".
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")

# Pre-initialize labeled synthetic metric so it appears in /metrics before any traffic.
from drift import SYNTH_REQUESTS_TOTAL
for _kind in ("normal", "drift"):
    for _status in ("success", "failure"):
        SYNTH_REQUESTS_TOTAL.labels(service=SERVICE_NAME, kind=_kind, status=_status).inc(0)



def _to_dict(pydantic_obj) -> dict:
    """Support Pydantic v2 (.model_dump) and v1 (.dict)."""
    if hasattr(pydantic_obj, "model_dump"):
        return pydantic_obj.model_dump()
    return pydantic_obj.dict()


def _extract_predicted_price(result: Any) -> float:
    """Extract numeric prediction from dict OR PredictionResponse."""
    if isinstance(result, dict):
        for key in ("price", "prediction", "predicted_price"):
            if key in result and result[key] is not None:
                return float(result[key])
        raise KeyError("Result dict missing prediction field (price|prediction|predicted_price).")

    for attr in ("price", "prediction", "predicted_price"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                return float(val)
    raise AttributeError("PredictionResponse missing price/prediction field.")


# -------------------------
# Routes
# -------------------------
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: HousePredictionRequest,
    x_synth: str | None = Header(default=None, alias="X-Synth"),
    x_synth_kind: str | None = Header(default=None, alias="X-Synth-Kind"),
):
    is_synth = (x_synth or "").strip() == "1"
    kind = (x_synth_kind or "normal").strip().lower()
    if kind not in ("normal", "drift"):
        kind = "normal"

    try:
        payload = _to_dict(request)
        result = predict_price(payload)  # returns dict

        model_requests_total.inc()

        # numeric features
        model_feature_numeric.labels("sqft").observe(float(request.sqft))
        model_feature_numeric.labels("bedrooms").observe(float(request.bedrooms))
        model_feature_numeric.labels("bathrooms").observe(float(request.bathrooms))
        model_feature_numeric.labels("year_built").observe(float(request.year_built))

        # categorical features
        model_feature_categorical_total.labels("location", str(request.location)).inc()
        model_feature_categorical_total.labels("condition", str(request.condition)).inc()

        # prediction distribution
        model_prediction_price.observe(_extract_predicted_price(result))

        if is_synth:
            record_synth_request(SERVICE_NAME, kind, "success")

        predicted = _extract_predicted_price(result)

        resp = PredictionResponse(
            predicted_price=predicted,
            confidence_interval=[predicted, predicted],
            features_importance={},
            prediction_time=datetime.utcnow().isoformat(),
        )
        return resp

    except Exception:
        if is_synth:
            record_synth_request(SERVICE_NAME, kind, "failure")
        raise


@app.post("/batch-predict", response_model=list[PredictionResponse])
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    payloads = [_to_dict(r) for r in requests]
    return batch_predict(payloads)
