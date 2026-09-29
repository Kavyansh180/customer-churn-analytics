"""End-to-end integration and edge-case testing for the complete ML system."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.evaluation.threshold_analysis import assign_churn_risk_band
from src.features.preprocessing import load_preprocessor
from src.models.model_utils import (
    DEFAULT_COEF_PATH,
    DEFAULT_FEATURE_NAMES_PATH,
    DEFAULT_LR_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    DEFAULT_RF_PATH,
    load_model,
)

client = TestClient(app)

BASE_VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12.0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.0,
    "TotalCharges": 660.0,
}


def test_end_to_end_inference_pipeline():
    """Verify end-to-end local inference: raw input -> pipeline -> LR model -> risk band."""
    preprocessor = load_preprocessor(DEFAULT_PREPROCESSOR_PATH)
    model = load_model(DEFAULT_LR_PATH)

    df_sample = pd.DataFrame([BASE_VALID_CUSTOMER])
    X_trans = preprocessor.transform(df_sample)

    assert X_trans.shape == (1, 47)
    prob = float(model.predict_proba(X_trans)[0, 1])
    assert 0.0 <= prob <= 1.0

    pred_class = int(prob >= 0.50)
    assert pred_class in [0, 1]

    risk_band = assign_churn_risk_band(prob)
    assert risk_band in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]


def test_fastapi_end_to_end_predict_endpoint():
    """Verify FastAPI /predict-churn endpoint end-to-end scoring."""
    payload = BASE_VALID_CUSTOMER.copy()
    payload["customerID"] = "INTEG-001"

    response = client.post("/predict-churn", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["customerID"] == "INTEG-001"
    assert "churn_probability" in data
    assert "predicted_churn" in data
    assert "risk_band" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_edge_case_minimum_values():
    """Verify edge-case minimum input (tenure=0, charges=0) executes safely."""
    min_payload = BASE_VALID_CUSTOMER.copy()
    min_payload["tenure"] = 0.0
    min_payload["MonthlyCharges"] = 0.0
    min_payload["TotalCharges"] = 0.0

    response = client.post("/predict-churn", json=min_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_band"] in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]


def test_edge_case_maximum_tenure():
    """Verify edge-case high-tenure customer (tenure=72 months) scores correctly."""
    high_payload = BASE_VALID_CUSTOMER.copy()
    high_payload["tenure"] = 72.0
    high_payload["Contract"] = "Two year"
    high_payload["MonthlyCharges"] = 115.0
    high_payload["TotalCharges"] = 8280.0

    response = client.post("/predict-churn", json=high_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    # High-tenure two-year contract accounts should score in Low Risk
    assert data["risk_band"] == "Low Risk"


def test_edge_case_unknown_categorical_handling():
    """Verify novel unseen categories are gracefully handled without error."""
    novel_payload = BASE_VALID_CUSTOMER.copy()
    novel_payload["PaymentMethod"] = "Digital Cryptocurrency (Novel)"
    novel_payload["InternetService"] = "Satellite Mesh (Novel)"

    response = client.post("/predict-churn", json=novel_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_edge_case_negative_tenure_validation():
    """Verify negative tenure triggers HTTP 422."""
    bad_payload = BASE_VALID_CUSTOMER.copy()
    bad_payload["tenure"] = -1.0
    response = client.post("/predict-churn", json=bad_payload)
    assert response.status_code == 422


def test_edge_case_negative_charges_validation():
    """Verify negative MonthlyCharges and TotalCharges trigger HTTP 422."""
    bad_payload1 = BASE_VALID_CUSTOMER.copy()
    bad_payload1["MonthlyCharges"] = -10.0
    assert client.post("/predict-churn", json=bad_payload1).status_code == 422

    bad_payload2 = BASE_VALID_CUSTOMER.copy()
    bad_payload2["TotalCharges"] = -50.0
    assert client.post("/predict-churn", json=bad_payload2).status_code == 422


def test_edge_case_missing_required_fields():
    """Verify omitting required fields (e.g. InternetService, Contract) triggers HTTP 422."""
    incomplete1 = BASE_VALID_CUSTOMER.copy()
    del incomplete1["InternetService"]
    assert client.post("/predict-churn", json=incomplete1).status_code == 422

    incomplete2 = BASE_VALID_CUSTOMER.copy()
    del incomplete2["Contract"]
    assert client.post("/predict-churn", json=incomplete2).status_code == 422


def test_deterministic_predictions():
    """Verify scoring the same customer profile repeatedly produces identical outputs."""
    res1 = client.post("/predict-churn", json=BASE_VALID_CUSTOMER).json()
    res2 = client.post("/predict-churn", json=BASE_VALID_CUSTOMER).json()
    res3 = client.post("/predict-churn", json=BASE_VALID_CUSTOMER).json()

    assert res1["churn_probability"] == res2["churn_probability"] == res3["churn_probability"]
    assert res1["predicted_churn"] == res2["predicted_churn"] == res3["predicted_churn"]
    assert res1["risk_band"] == res2["risk_band"] == res3["risk_band"]


def test_all_persisted_artifacts_integrity():
    """Verify presence and valid structure of all core serialized artifacts."""
    artifacts = [
        DEFAULT_PREPROCESSOR_PATH,
        DEFAULT_LR_PATH,
        DEFAULT_RF_PATH,
        DEFAULT_COEF_PATH,
        DEFAULT_METADATA_PATH,
        DEFAULT_FEATURE_NAMES_PATH,
    ]
    for p in artifacts:
        assert p.exists(), f"Artifact missing: {p}"
        assert p.stat().st_size > 0

    df_names = pd.read_csv(DEFAULT_FEATURE_NAMES_PATH)
    assert len(df_names) == 47

    df_coef = pd.read_csv(DEFAULT_COEF_PATH)
    assert len(df_coef) == 47


@pytest.mark.parametrize(
    "internet,contract,payment,has_phone",
    [
        ("Fiber optic", "Month-to-month", "Electronic check", "Yes"),
        ("DSL", "One year", "Bank transfer (automatic)", "No"),
        ("No", "Two year", "Mailed check", "Yes"),
        ("DSL", "Two year", "Credit card (automatic)", "Yes"),
    ],
)
def test_various_categorical_combinations(internet, contract, payment, has_phone):
    """Verify that varied categorical profile combinations transform and predict correctly."""
    payload = BASE_VALID_CUSTOMER.copy()
    payload["InternetService"] = internet
    payload["Contract"] = contract
    payload["PaymentMethod"] = payment
    payload["PhoneService"] = has_phone
    if has_phone == "No":
        payload["MultipleLines"] = "No phone service"
    if internet == "No":
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            payload[col] = "No internet service"

    res = client.post("/predict-churn", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["predicted_churn"] in [0, 1]
    assert data["risk_band"] in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]


@pytest.mark.parametrize(
    "prob,expected_band",
    [
        (0.0, "Low Risk"),
        (0.1999, "Low Risk"),
        (0.20, "Medium Risk"),
        (0.4999, "Medium Risk"),
        (0.50, "High Risk"),
        (0.7499, "High Risk"),
        (0.75, "Critical Risk"),
        (1.0, "Critical Risk"),
    ],
)
def test_risk_band_exact_boundaries(prob, expected_band):
    """Verify exact risk band assignment according to documented cutoffs."""
    assert assign_churn_risk_band(prob) == expected_band


def test_reference_validation_profiles():
    """Verify the 3 reference benchmark customer profiles from Phase 7 validation."""
    low_profile = {
        "customerID": "CUST-LOW-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.00,
        "TotalCharges": 3900.00,
    }
    med_profile = {
        "customerID": "CUST-MED-002",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24.0,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 80.00,
        "TotalCharges": 1920.00,
    }
    high_profile = {
        "customerID": "CUST-HIGH-003",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2.0,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.00,
        "TotalCharges": 190.00,
    }

    res_low = client.post("/predict-churn", json=low_profile).json()
    assert abs(res_low["churn_probability"] - 0.0035) < 0.001
    assert res_low["risk_band"] == "Low Risk"
    assert res_low["predicted_churn"] == 0

    res_med = client.post("/predict-churn", json=med_profile).json()
    assert abs(res_med["churn_probability"] - 0.2863) < 0.001
    assert res_med["risk_band"] == "Medium Risk"
    assert res_med["predicted_churn"] == 0

    res_high = client.post("/predict-churn", json=high_profile).json()
    assert abs(res_high["churn_probability"] - 0.8513) < 0.001
    assert res_high["risk_band"] == "Critical Risk"
    assert res_high["predicted_churn"] == 1



