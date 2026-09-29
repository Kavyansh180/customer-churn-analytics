"""Pydantic validation schemas for Customer Churn Prediction API."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerInputSchema(BaseModel):
    """Input customer feature schema for churn prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )

    customerID: Optional[str] = Field(
        default=None,
        description="Optional customer identifier. Never used as a predictive model feature.",
    )
    gender: str = Field(..., description="Customer gender (e.g., 'Female', 'Male')")
    SeniorCitizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether customer is a senior citizen (1: Yes, 0: No)",
    )
    Partner: str = Field(..., description="Whether customer has a partner ('Yes', 'No')")
    Dependents: str = Field(..., description="Whether customer has dependents ('Yes', 'No')")
    tenure: float = Field(
        ...,
        ge=0.0,
        description="Number of months customer has stayed with the company",
    )
    PhoneService: str = Field(..., description="Whether customer has phone service ('Yes', 'No')")
    MultipleLines: str = Field(
        ...,
        description="Whether customer has multiple lines ('Yes', 'No', 'No phone service')",
    )
    InternetService: str = Field(
        ...,
        description="Customer's internet service provider ('DSL', 'Fiber optic', 'No')",
    )
    OnlineSecurity: str = Field(
        ...,
        description="Whether customer has online security ('Yes', 'No', 'No internet service')",
    )
    OnlineBackup: str = Field(
        ...,
        description="Whether customer has online backup ('Yes', 'No', 'No internet service')",
    )
    DeviceProtection: str = Field(
        ...,
        description="Whether customer has device protection ('Yes', 'No', 'No internet service')",
    )
    TechSupport: str = Field(
        ...,
        description="Whether customer has tech support ('Yes', 'No', 'No internet service')",
    )
    StreamingTV: str = Field(
        ...,
        description="Whether customer has streaming TV ('Yes', 'No', 'No internet service')",
    )
    StreamingMovies: str = Field(
        ...,
        description="Whether customer has streaming movies ('Yes', 'No', 'No internet service')",
    )
    Contract: str = Field(
        ...,
        description="Contract term ('Month-to-month', 'One year', 'Two year')",
    )
    PaperlessBilling: str = Field(
        ...,
        description="Whether customer has paperless billing ('Yes', 'No')",
    )
    PaymentMethod: str = Field(
        ...,
        description="Payment method (e.g., 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')",
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        description="Amount charged to customer monthly in USD",
    )
    TotalCharges: float = Field(
        ...,
        ge=0.0,
        description="Total amount charged to customer over lifetime in USD",
    )


class PredictionResponse(BaseModel):
    """Response payload for churn probability and risk tier prediction."""

    customerID: Optional[str] = Field(default=None, description="Echoed customer identifier if provided")
    churn_probability: float = Field(..., description="Estimated churn probability in [0.0, 1.0]")
    predicted_churn: int = Field(..., description="Binary churn prediction (1: Churn, 0: Retain)")
    risk_band: str = Field(..., description="Operational risk tier: Low Risk, Medium Risk, High Risk, Critical Risk")
    decision_threshold: float = Field(default=0.50, description="Decision boundary threshold applied")


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = Field(default="healthy", description="Service health indicator")


class ModelInfoResponse(BaseModel):
    """Model information and metadata response."""

    model: str = Field(..., description="Name of the deployed predictive model")
    model_version: str = Field(default="1.0.0", description="Model version tag")
    model_role: str = Field(default="Primary Interpretable Model", description="Model role in analytics architecture")
    feature_count: int = Field(..., description="Total transformed features ingested by model")
    decision_threshold: float = Field(default=0.50, description="Default classification boundary")
    risk_bands: Dict[str, str] = Field(..., description="Operational churn risk tier probability boundaries")
    training_summary: Optional[Dict[str, Any]] = Field(default=None, description="High-level training metadata")
