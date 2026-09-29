"""FastAPI application package for Customer Churn Analytics."""

from src.api.main import app
from src.api.schemas import (
    CustomerInputSchema,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.api.service import ChurnPredictionService, get_prediction_service

__all__ = [
    "app",
    "CustomerInputSchema",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictionResponse",
    "ChurnPredictionService",
    "get_prediction_service",
]
