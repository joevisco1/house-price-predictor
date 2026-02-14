from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server
import threading

# ✅ drain flag (module-level)
DRAINING = False

app = FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features. "
        "This application is part of the MLOps Bootcamp by School of Devops. "
        "Authored by Gourav Shah."
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

Instrumentator().instrument(app).expose(app)

def start_metrics_server():
    start_http_server(9100)

threading.Thread(target=start_metrics_server, daemon=True).start()

# ✅ keep /health for humans / basic checks (optional)
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "draining": DRAINING}

# ✅ probes (these match your rollout hardening)
@app.get("/health/live")
async def health_live():
    return {"status": "live"}

@app.get("/health/ready")
async def health_ready(response: Response):
    if DRAINING:
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "draining"}
    return {"status": "ready"}

@app.post("/health/drain")
async def health_drain():
    global DRAINING
    DRAINING = True
    return {"status": "draining_started"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    return predict_price(request)

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list[PredictionResponse])
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    return batch_predict(requests)
