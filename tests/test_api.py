"""Unit tests for Phase 7: FastAPI Model Serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_CUSTOMER_PAYLOAD = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1.0,
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


def test_fastapi_app_imports_successfully():
    """1. Verify FastAPI app instance exists and has configured routes."""
    assert app is not None
    assert app.title == "Customer Churn Prediction & Retention Analytics API"


def test_health_endpoint_status_code():
    """2. Verify /health returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_payload():
    """3. Verify /health returns expected status body."""
    response = client.get("/health")
    data = response.json()
    assert data == {"status": "healthy"}


def test_model_info_status_code():
    """4. Verify /model-info returns HTTP 200 OK."""
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_reports_47_features():
    """5. Verify /model-info correctly reports 47 transformed features."""
    response = client.get("/model-info")
    data = response.json()
    assert data["model"] == "Logistic Regression"
    assert data["feature_count"] == 47
    assert data["decision_threshold"] == 0.50
    assert "risk_bands" in data


def test_predict_churn_valid_request_returns_200():
    """6. Verify valid /predict-churn request returns HTTP 200 OK."""
    response = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD)
    assert response.status_code == 200


def test_predict_churn_response_structure():
    """7. Verify response contains churn_probability and echoed customerID."""
    response = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD)
    data = response.json()
    assert "churn_probability" in data
    assert "predicted_churn" in data
    assert "risk_band" in data
    assert data["customerID"] == "7590-VHVEG"


def test_predict_churn_probability_in_zero_one_range():
    """8. Verify churn probability is strictly within [0.0, 1.0]."""
    response = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD)
    data = response.json()
    prob = data["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_churn_class_is_binary():
    """9. Verify predicted_churn is either 0 or 1."""
    response = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD)
    data = response.json()
    assert data["predicted_churn"] in [0, 1]


def test_predict_churn_risk_band_validity():
    """10. Verify assigned risk band belongs to the 4 defined operational categories."""
    valid_bands = {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}
    response = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD)
    data = response.json()
    assert data["risk_band"] in valid_bands


def test_invalid_negative_tenure_returns_422():
    """11. Verify negative tenure is rejected with HTTP 422 Unprocessable Entity."""
    invalid_payload = VALID_CUSTOMER_PAYLOAD.copy()
    invalid_payload["tenure"] = -5.0
    response = client.post("/predict-churn", json=invalid_payload)
    assert response.status_code == 422


def test_invalid_negative_monthly_charges_returns_422():
    """12. Verify negative MonthlyCharges is rejected with HTTP 422."""
    invalid_payload = VALID_CUSTOMER_PAYLOAD.copy()
    invalid_payload["MonthlyCharges"] = -29.85
    response = client.post("/predict-churn", json=invalid_payload)
    assert response.status_code == 422


def test_missing_required_field_returns_422():
    """13. Verify omitting a mandatory field (e.g., Contract) returns HTTP 422."""
    invalid_payload = VALID_CUSTOMER_PAYLOAD.copy()
    del invalid_payload["Contract"]
    response = client.post("/predict-churn", json=invalid_payload)
    assert response.status_code == 422


def test_unknown_categorical_value_handled_gracefully():
    """14. Verify novel unseen categories are encoded smoothly by the pipeline."""
    novel_payload = VALID_CUSTOMER_PAYLOAD.copy()
    novel_payload["PaymentMethod"] = "Crypto Payment (Unseen)"
    novel_payload["InternetService"] = "Satellite (Unseen)"
    response = client.post("/predict-churn", json=novel_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_prediction_is_deterministic():
    """15. Verify identical inputs produce identical probabilities and predictions."""
    res1 = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD).json()
    res2 = client.post("/predict-churn", json=VALID_CUSTOMER_PAYLOAD).json()
    assert res1["churn_probability"] == res2["churn_probability"]
    assert res1["predicted_churn"] == res2["predicted_churn"]
    assert res1["risk_band"] == res2["risk_band"]
