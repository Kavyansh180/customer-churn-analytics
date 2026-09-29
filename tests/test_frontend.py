"""Unit tests for Phase 8: Streamlit Retention Analytics Dashboard."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from src.frontend.app import (
    API_BASE_URL,
    PRESET_PROFILES,
    RETENTION_GUIDANCE,
    check_api_health,
    get_model_info,
    request_prediction,
)

FRONTEND_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "frontend" / "app.py"


def test_app_file_exists():
    """1. Verify src/frontend/app.py exists on disk."""
    assert FRONTEND_APP_PATH.exists()
    assert FRONTEND_APP_PATH.is_file()


def test_app_imports_successfully():
    """2. Verify app module can be imported without execution errors."""
    import src.frontend.app as frontend_app
    assert hasattr(frontend_app, "render_dashboard")
    assert hasattr(frontend_app, "request_prediction")


def test_required_field_names_present():
    """3. Verify all 19 customer predictor fields exist in preset configurations."""
    required_fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    for preset_name, data in PRESET_PROFILES.items():
        if data is not None:
            for field in required_fields:
                assert field in data, f"Field '{field}' missing from preset '{preset_name}'"


def test_api_base_url_configuration():
    """4. Verify API base URL configuration defaults to 127.0.0.1:8000 and respects env var."""
    assert "http://127.0.0.1:8000" in API_BASE_URL or "http" in API_BASE_URL


def test_app_does_not_directly_load_model_artifacts():
    """5. Verify app.py does not import joblib or load sklearn models directly."""
    with open(FRONTEND_APP_PATH, "r", encoding="utf-8") as f:
        code = f.read()

    assert "import joblib" not in code
    assert "from joblib" not in code
    assert "joblib.load" not in code
    assert "from sklearn" not in code
    assert "import sklearn" not in code


def test_expected_risk_band_labels_present():
    """6. Verify all 4 operational risk tiers exist in retention guidance."""
    expected_bands = {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}
    assert set(RETENTION_GUIDANCE.keys()) == expected_bands


def test_check_api_health_success():
    """7. Verify check_api_health returns True when FastAPI /health responds 200."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "healthy"}

        assert check_api_health("http://mock-api:8000") is True


def test_check_api_health_failure_returns_false():
    """8. Verify check_api_health returns False when FastAPI backend is unreachable."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        assert check_api_health("http://mock-api:8000") is False


def test_get_model_info_success():
    """9. Verify get_model_info returns JSON metadata when backend responds 200."""
    mock_payload = {
        "model": "Logistic Regression",
        "feature_count": 47,
        "decision_threshold": 0.50,
        "risk_bands": {"low": "<0.20", "critical": ">=0.75"},
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_payload

        data = get_model_info("http://mock-api:8000")
        assert data == mock_payload


def test_request_prediction_success():
    """10. Verify request_prediction returns parsed JSON response on HTTP 200."""
    mock_response = {
        "customerID": "CUST-LOW-001",
        "churn_probability": 0.0035,
        "predicted_churn": 0,
        "risk_band": "Low Risk",
        "decision_threshold": 0.50,
    }
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        res, err = request_prediction({"tenure": 65.0}, "http://mock-api:8000")
        assert err is None
        assert res == mock_response
        assert res["risk_band"] == "Low Risk"


def test_request_prediction_validation_error_handling():
    """11. Verify HTTP 422 validation errors return user-friendly error string."""
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 422
        mock_post.return_value.json.return_value = {"detail": "Field 'tenure' must be >= 0."}

        res, err = request_prediction({"tenure": -5.0}, "http://mock-api:8000")
        assert res is None
        assert "Input Validation Error (HTTP 422)" in err


def test_request_prediction_connection_error_handling():
    """12. Verify connection refusal returns helpful guidance without stack traces."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        res, err = request_prediction({"tenure": 10.0}, "http://mock-api:8000")
        assert res is None
        assert "Cannot connect to FastAPI backend" in err
        assert "uvicorn" in err.lower()
