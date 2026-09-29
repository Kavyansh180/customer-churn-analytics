# Streamlit Retention Analytics Dashboard Report
**Project:** Customer Churn Prediction & Retention Analytics System  
**Frontend Framework:** Streamlit 1.64.0  
**Backend API:** FastAPI 0.139.0 / Uvicorn ASGI Server  
**Scope:** Phase 8 — End-to-End Retention Analytics Web Interface  

---

## 1. Dashboard Purpose & Strategic Role

The **Customer Churn Prediction & Retention Analytics Dashboard** provides an interactive, operational user interface designed for customer relationship managers, retention specialists, and business analysts. 

### Core Capabilities:
- **Interactive Scoring:** Allows business users to adjust customer account configurations, contract terms, subscribed features, and monthly spend to evaluate churn probability in real time.
- **Operational Risk Segmentation:** Classifies accounts into 4 operational tiers (**Low, Medium, High, Critical Risk**) to guide targeted retention interventions.
- **Decoupled Architecture:** Operates strictly as a frontend consumer of the **FastAPI model-serving layer**, maintaining clean enterprise separation between UI and model runtime.

---

## 2. System Architecture & Communication Flow

```
+───────────────────────────+
│   Customer Retention User │
+─────────────┬─────────────+
              │ Web Browser
              ▼
+───────────────────────────+
│   Streamlit Dashboard     │  (src/frontend/app.py)
│  (UI Form / Metric Cards) │
+─────────────┬─────────────+
              │ HTTP POST /predict-churn (JSON Payload)
              ▼
+───────────────────────────+
│   FastAPI Serving Layer   │  (src/api/main.py)
+─────────────┬─────────────+
              │
              ├── Preprocessing Pipeline (artifacts/preprocessor.joblib)
              │     - Calculates total_services & monthly_charges_diff
              │     - Standardizes & One-Hot Encodes ──> (1, 47) array
              │
              └── Logistic Regression Model (artifacts/logistic_regression.joblib)
                    - Scores probability P(Churn=1 | X)
                    - Applies 0.50 Decision Threshold
                    - Maps to Operational Churn Risk Band
              │
              ▼ HTTP 200 OK (PredictionResponse JSON)
+───────────────────────────+
│   Streamlit Display Card  │
│ - 85.13% Churn Probability│
│ - Likely to Churn         │
│ - Critical Risk Band      │
│ - Actionable Strategy     │
+───────────────────────────+
```

---

## 3. Key Frontend Features & UI Components

### A. Dynamic API Health & Model Info Card
- The dashboard automatically pings `GET /health` and `GET /model-info` on startup.
- Displays connection status badge (**"🟢 API Status: Connected"** or **"🔴 API Status: Unavailable"**).
- When the backend is offline, shows user-friendly troubleshooting instructions rather than raw Python stack traces.

### B. Demo Customer Presets
To facilitate rapid testing during live demonstrations and interviews, a preset selector auto-populates the input form with three validated test profiles:
1. **Low-Risk Loyal Customer Profile:** ($p \approx 0.35\%$, Low Risk)
2. **Moderate-Risk Intermediate Profile:** ($p \approx 28.63\%$, Medium Risk)
3. **High-Risk Attrition Profile:** ($p \approx 85.13\%$, Critical Risk)

### C. Comprehensive Customer Profile Form
Collects all 19 predictive attributes categorized into three logical visual columns:
- **Column 1:** Account & Demographics (`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`).
- **Column 2:** Contract & Billing (`Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges`).
- **Column 3:** Subscribed Services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).

### D. Prediction Metrics & Strategic Action Guidance
Upon clicking **"🎯 Predict Churn Risk"**, the dashboard displays:
- **Churn Probability Metric:** Formatted percentage with delta relative to the 50% decision boundary.
- **Predicted Class:** "Likely to Churn" (1) vs. "Likely to Retain" (0).
- **Assigned Risk Tier Alert:** Color-coded alert box (Green = Low, Blue = Medium, Amber = High, Red = Critical).
- **Retention Decision Support:** Contextual business recommendation corresponding to the predicted risk tier.

---

## 4. Operational Risk Bands & Strategic Guidance

| Operational Risk Band | Probability Threshold | Strategic Orientation | Prescribed Business Action |
| :--- | :---: | :--- | :--- |
| 🟢 **Low Risk** | $< 20.0\%$ | General Relationship Care | Standard marketing newsletters; no retention spend required. |
| 🟡 **Medium Risk** | $20.0\% - < 50.0\%$ | Early Adoption & Engagement | Feature onboarding prompts, survey check-ins, product tips. |
| 🟠 **High Risk** | $50.0\% - < 75.0\%$ | Proactive Outreach | Account manager check-in, 1-year contract incentive offer. |
| 🔴 **Critical Risk** | $\ge 75.0\%$ | Emergency Intervention | High-priority outreach, contract renewal credit, executive review. |

---

## 5. Error Handling & Edge-Case Resilience

1. **FastAPI Server Downtime:** Gracefully caught via `requests.exceptions.ConnectionError`; displays actionable prompt to launch Uvicorn server.
2. **Network Timeout:** 5-second request timeout prevents UI hangs.
3. **Validation Errors (HTTP 422):** Formats Pydantic validation errors cleanly into `st.error` notifications.
4. **Information Security:** Never exposes local directory paths, traceback strings, or server internals to end users.

---

## 6. How to Run the End-to-End Application

### Step 1: Start FastAPI Model Serving Backend (Terminal 1)
```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Launch Streamlit Dashboard (Terminal 2)
```powershell
streamlit run src/frontend/app.py
```
*Access the dashboard at `http://localhost:8501`.*
