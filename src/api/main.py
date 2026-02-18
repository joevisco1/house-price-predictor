from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

import os

from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# NEW: synthetic traffic metric helper (added in drift.py)
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
# Drift / model metrics
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

# NEW: label value for synthetic traffic metric
# Set SERVICE_NAME in the container env to "model-active" or "model-preview".
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")


def _extract_predicted_price(result: PredictionResponse) -> float:
    """
    Robust extraction so we don't guess your field name.
    Common patterns: result.price, result.prediction, result.predicted_price.
    """
    for attr in ("price", "prediction", "predicted_price"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                return float(val)
    raise AttributeError(
        "PredictionResponse does not have a recognizable price field. "
        "Expected one of: price, prediction, predicted_price."
    )


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
    """
    If the CronJob sends:
      X-Synth: 1
      X-Synth-Kind: normal|drift
    then we increment:
      model_synthetic_requests_total{service,kind,status}
    """
    is_synth = (x_synth or "").strip() == "1"
    kind = (x_synth_kind or "normal").strip().lower()
    if kind not in ("normal", "drift"):
        kind = "normal"

    try:
        result = predict_price(request)

        # request counter
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

        # NEW: synthetic success metric
        if is_synth:
            record_synth_request(SERVICE_NAME, kind, "success")

        return result

    except Exception:
        # NEW: synthetic failure metric (only for exceptions inside the handler)
        if is_synth:
            record_synth_request(SERVICE_NAME, kind, "failure")
        raise


@app.post("/batch-predict", response_model=list)
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    return batch_predict(requests)
