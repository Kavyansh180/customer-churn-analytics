# FastAPI Model Serving & REST API Report
**Project:** Customer Churn Prediction & Retention Analytics System  
**Framework:** FastAPI 0.139.0 / Pydantic v2.13.4 / Uvicorn ASGI Server  
**Artifacts Reused:** `artifacts/preprocessor.joblib`, `artifacts/logistic_regression.joblib`, `artifacts/model_metadata.json`  
**Scope:** Phase 7 — Production Model Serving  

---

## 1. API Architecture & Layered Design

The API service follows a strict separation-of-concerns pattern:

```
src/api/
├── main.py          # FastAPI application, route handlers, error wrappers, lifespan events
├── schemas.py       # Pydantic v2 validation models (CustomerInputSchema, PredictionResponse)
├── service.py       # ChurnPredictionService singleton, artifact caching, inference pipeline
└── __init__.py      # Package interface
```

### Request Lifecycle Architecture

```
HTTP POST /predict-churn (JSON Payload)
       │
       ▼
[ Pydantic Validation (CustomerInputSchema) ] ── (Invalid) ──> HTTP 422 Unprocessable Entity
       │ (Valid)
       ▼
[ ChurnPredictionService.predict() ]
       │
       ├── Extract optional customerID (Identifier only; never passed to model)
       ├── Build 1-row Pandas DataFrame (19 predictor features)
       ├── FeatureEngineer + ColumnTransformer (artifacts/preprocessor.joblib) ──> (1, 47) array
       ├── Logistic Regression Scoring (artifacts/logistic_regression.joblib)
       ├── Compute P(Churn=1 | X) via predict_proba
       ├── Apply 0.50 Decision Boundary ──> predicted_churn (0 or 1)
       └── Assign Operational Risk Tier ──> risk_band (Low, Medium, High, Critical)
       │
       ▼
HTTP 200 OK (PredictionResponse JSON)
```

---

## 2. Why Model Artifacts are Loaded at Startup (Not Retrained)

1. **Sub-Millisecond Inference Latency:** Pre-loading serialized models into memory during FastAPI startup lifespan enables scoring in under 5 milliseconds per request, rather than seconds/minutes required for training.
2. **Deterministic Reproducibility:** Serving relies on frozen, version-controlled parameters learned during Phase 4 and Phase 5.
3. **Architectural Decoupling:** Separates offline batch training jobs from online real-time serving infrastructure.

---

## 3. Pydantic Input Validation & Domain Constraints

The API strictly validates all incoming JSON fields before executing feature transformations:

| Field Name | Expected Type | Validation Rule | Business / Domain Reason |
| :--- | :---: | :---: | :--- |
| `SeniorCitizen` | `int` | `ge=0, le=1` | Must be binary flag (0: No, 1: Yes). |
| `tenure` | `float / int` | `ge=0.0` | Customer account lifetime cannot be negative. |
| `MonthlyCharges` | `float` | `ge=0.0` | Recurring monthly bill must be non-negative. |
| `TotalCharges` | `float` | `ge=0.0` | Cumulative spend must be non-negative. |
| `Contract` | `str` | Required | Critical churn driver ('Month-to-month', 'One year', 'Two year'). |
| `customerID` | `Optional[str]` | Optional | Client tracking identifier; strictly stripped before scoring. |

*Invalid inputs automatically yield `HTTP 422 Unprocessable Entity` with clear field-level validation errors.*

---

## 4. End-to-End Pipeline & Feature Engineering Consistency

- **Zero Duplication:** The API does not manually re-implement feature formulas or scaling equations in Python code.
- **Pre-trained Pipeline Execution:** The incoming dictionary is passed directly through `artifacts/preprocessor.joblib`, which automatically:
  1. Calculates `total_services` (0 to 8 count of active add-on features).
  2. Calculates `monthly_charges_diff` ($\text{MonthlyCharges} - \frac{\text{TotalCharges}}{\max(\text{tenure}, 1.0)}$).
  3. Imputes missing values using training medians and modes.
  4. Standardizes numericals using training $\mu$ and $\sigma$.
  5. One-hot encodes categoricals using training vocabularies (`handle_unknown='ignore'`).

---

## 5. Decision Boundary & Operational Risk Banding

- **Threshold Assignment:** The default classification decision uses $0.50$:
  $$\hat{y} = \mathbb{I}(\hat{p} \ge 0.50)$$
- **Operational Churn Risk Tiers:**
  - **`Low Risk` ($p < 0.20$):** Retain baseline service; no discount allocation.
  - **`Medium Risk` ($0.20 \le p < 0.50$):** Moderate risk; trigger automated engagement prompts.
  - **`High Risk` ($0.50 \le p < 0.75$):** Elevated risk; assign to proactive retention queue.
  - **`Critical Risk` ($p \ge 0.75$):** Imminent attrition; offer contract extension incentive.

---

## 6. Real-World API Scenarios & Actual Verification Results

### Scenario A: Low-Risk Loyal Customer Profile
- **Input Attributes:** 2-year contract, 65-month tenure, DSL internet, active tech support & backup, bank auto-pay, $60.00/mo.
- **Actual API Output:**
```json
{
  "customerID": "CUST-LOW-001",
  "churn_probability": 0.0035,
  "predicted_churn": 0,
  "risk_band": "Low Risk",
  "decision_threshold": 0.5
}
```

### Scenario B: Moderate-Risk Intermediate Customer Profile
- **Input Attributes:** 1-year contract, 24-month tenure, fiber optic, paperless billing, credit card auto-pay, $80.00/mo.
- **Actual API Output:**
```json
{
  "customerID": "CUST-MED-002",
  "churn_probability": 0.2863,
  "predicted_churn": 0,
  "risk_band": "Medium Risk",
  "decision_threshold": 0.5
}
```

### Scenario C: High-Risk Attrition Customer Profile
- **Input Attributes:** Month-to-month contract, 2-month tenure, senior citizen, fiber optic, electronic check, no security/support, $95.00/mo.
- **Actual API Output:**
```json
{
  "customerID": "CUST-HIGH-003",
  "churn_probability": 0.8513,
  "predicted_churn": 1,
  "risk_band": "Critical Risk",
  "decision_threshold": 0.5
}
```

---

## 7. Interactive OpenAPI & Swagger Documentation

- **Interactive Swagger UI:** Accessible at `http://localhost:8000/docs`
- **ReDoc Interactive Documentation:** Accessible at `http://localhost:8000/redoc`
- **Machine-Readable OpenAPI Specification:** Exported at `http://localhost:8000/openapi.json`

---

## 8. Local Server Execution Instructions

Start the production-ready ASGI server via Uvicorn:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
