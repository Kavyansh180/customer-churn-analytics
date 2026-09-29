"""FastAPI application for Customer Churn Prediction & Retention Analytics System."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    CustomerInputSchema,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.api.service import get_prediction_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to pre-load model artifacts at startup."""
    try:
        # Preload singleton service and verify artifact loading
        service = get_prediction_service()
    except Exception as exc:
        print(f"CRITICAL: Failed to load model artifacts during startup: {exc}")
    yield


app = FastAPI(
    title="Customer Churn Prediction & Retention Analytics API",
    description=(
        "Production-grade REST API serving calibrated customer churn probabilities, "
        "classification decisions, and operational churn risk tiers for retention workflows."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["Health & Status"],
)
async def health_check() -> HealthResponse:
    """Return operational health status of the API service."""
    return HealthResponse(status="healthy")


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Risk Band Definitions",
    tags=["Model Information"],
)
async def get_model_info() -> ModelInfoResponse:
    """Return model specifications, feature dimensions, decision threshold, and risk tier definitions."""
    try:
        service = get_prediction_service()
        return service.get_model_information()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model metadata.",
        ) from exc


@app.post(
    "/predict-churn",
    response_model=PredictionResponse,
    summary="Predict Customer Churn Probability & Risk Tier",
    tags=["Predictions"],
)
async def predict_churn(customer_data: CustomerInputSchema) -> PredictionResponse:
    """Score an individual customer profile against the trained Logistic Regression model.

    Returns estimated churn probability, binary prediction at 0.50 threshold,
    and assigned operational risk band.
    """
    try:
        service = get_prediction_service()
        return service.predict(customer_data)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid data encountered during transformation: {str(val_err)}",
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during prediction inference.",
        ) from exc
