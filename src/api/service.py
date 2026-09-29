"""Model loading and prediction service layer for FastAPI application."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.schemas import CustomerInputSchema, ModelInfoResponse, PredictionResponse
from src.evaluation.threshold_analysis import assign_churn_risk_band
from src.features.preprocessing import load_preprocessor
from src.models.model_utils import (
    DEFAULT_LR_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    load_model,
)

PREDICTOR_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class ChurnPredictionService:
    """Service class managing preprocessor, model artifact lifecycle, and real-time inference."""

    def __init__(
        self,
        preprocessor_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.preprocessor_path = preprocessor_path or DEFAULT_PREPROCESSOR_PATH
        self.model_path = model_path or DEFAULT_LR_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH

        self.preprocessor: Optional[Pipeline] = None
        self.model: Optional[LogisticRegression] = None
        self.metadata: Dict[str, Any] = {}

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load and cache the preprocessor and Logistic Regression model at initialization."""
        try:
            self.preprocessor = load_preprocessor(self.preprocessor_path)
            self.model = load_model(self.model_path)

            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {str(e)}") from e

    def predict(self, customer: CustomerInputSchema) -> PredictionResponse:
        """Score a single customer profile through the preprocessor and Logistic Regression model.

        Workflow:
        1. Parse Pydantic schema to dictionary and extract optional customerID.
        2. Format single-row DataFrame using exact predictor columns (excluding customerID).
        3. Transform raw row using the Phase 4 Scikit-Learn Pipeline.
        4. Calculate predicted churn probability P(Churn=1 | X).
        5. Apply 0.50 classification threshold and assign operational risk tier.

        Args:
            customer: Validated CustomerInputSchema instance.

        Returns:
            PredictionResponse containing probability, class, and risk tier.
        """
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Model service artifacts are not initialized.")

        # 1. Convert to dictionary
        data_dict = customer.model_dump()
        customer_id = data_dict.get("customerID")

        # 2. Extract strictly predictive columns (never customerID)
        row_dict = {col: data_dict[col] for col in PREDICTOR_FEATURE_COLUMNS if col in data_dict}
        df_row = pd.DataFrame([row_dict])

        # 3. Transform via loaded preprocessor pipeline
        X_trans = self.preprocessor.transform(df_row)

        # 4. Predict probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])

        # 5. Apply threshold and risk tier
        decision_thresh = 0.50
        pred_class = int(prob >= decision_thresh)
        risk_tier = assign_churn_risk_band(prob)

        return PredictionResponse(
            customerID=customer_id,
            churn_probability=round(prob, 4),
            predicted_churn=pred_class,
            risk_band=risk_tier,
            decision_threshold=decision_thresh,
        )

    def get_model_information(self) -> ModelInfoResponse:
        """Retrieve deployment metadata and operational risk band definitions."""
        feature_count = 47
        if self.preprocessor is not None:
            try:
                col_transformer = self.preprocessor.named_steps.get("preprocessor")
                if col_transformer:
                    feature_count = len(col_transformer.get_feature_names_out())
            except Exception:
                pass

        return ModelInfoResponse(
            model="Logistic Regression",
            model_version="1.0.0",
            model_role="Primary Interpretable Model",
            feature_count=feature_count,
            decision_threshold=0.50,
            risk_bands={
                "low": "<0.20",
                "medium": "0.20-<0.50",
                "high": "0.50-<0.75",
                "critical": ">=0.75",
            },
            training_summary=self.metadata.get("dataset_dimensions"),
        )


# Global singleton service instance
_service_instance: Optional[ChurnPredictionService] = None


def get_prediction_service() -> ChurnPredictionService:
    """Access singleton ChurnPredictionService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ChurnPredictionService()
    return _service_instance
