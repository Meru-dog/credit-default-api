"""FastAPI service for the fixed credit default model."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model import load_model
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("credit_default_api")

LIMITATIONS = [
    "Learning demo only. Do not submit real customer data.",
    "Not for real credit decisions.",
    "The threshold is based on an assumed 5:1 cost ratio.",
    "Trained on 2005 Taiwanese credit card data.",
    "Excluding demographic fields does not establish fairness.",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # A failure here stops startup, so an invalid model is never served.
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Credit Default Model API",
    version="1.0.0",
    description=(
        "Learning demo that flags records for human review. "
        "Do not submit real customer data. Not for real credit decisions."
    ),
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    # The default response echoes input values and fails on NaN or Infinity.
    details = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.middleware("http")
async def log_request(request: Request, call_next) -> Response:
    started = time.perf_counter()
    response = await call_next(request)
    model = getattr(request.app.state, "model", None)
    # Request bodies and input values are deliberately not logged.
    logger.info(
        "time=%s path=%s status=%s duration_ms=%.1f model_version=%s",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
        model.version if model else "unavailable",
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", model_version=request.app.state.model.version)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    model = request.app.state.model
    return ModelInfoResponse(
        model_version=model.version,
        threshold=model.threshold,
        features=model.features,
        limitations=LIMITATIONS,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    model = request.app.state.model
    probability, review = model.predict(payload)
    return PredictionResponse(
        default_probability=probability,
        review=review,
        threshold=model.threshold,
        model_version=model.version,
    )
