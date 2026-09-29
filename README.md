# Customer Churn Prediction & Retention Analytics System

An end-to-end decision-support prototype for customer churn risk prediction, statistical attribution, operational risk banding, and retention decision support via a high-performance REST API and interactive dashboard.

---

## 1. Project Title
**Customer Churn Prediction & Retention Analytics System**

---

## 2. Problem Statement
In the subscription telecommunications sector, acquiring new customers costs significantly more than retaining existing accounts. Unanticipated customer attrition erodes recurring revenue, lowers Customer Lifetime Value (LTV), and elevates replacement acquisition expenditures. By identifying early warning signals (e.g., month-to-month contracts, fiber optic pricing friction, manual electronic check payments), organizations can deploy timely, targeted retention interventions before accounts cancel their service.

---

## 3. Objective
To provide decision-support capabilities for retention teams by generating probabilistic churn scores, transparent statistical feature attribution via odds ratios, decision threshold tuning, and actionable operational risk tiers through a modular, reproducible, and leakage-safe machine learning workflow.

---

## 4. Dataset
- **Source:** IBM Telco Customer Churn sample dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
  - *Note:* The dataset is included for reproducibility in this repository. Users should verify the applicable terms from the original dataset source before redistribution or commercial use.
- **Raw Dimensions:** 7,043 rows × 21 columns
- **Cleaned Dimensions:** 7,032 rows × 21 columns
- **Target Variable:** `Churn` (`No` → 0 [73.42%], `Yes` → 1 [26.58%])
- **Attribute Categories:**
  - **Identifiers:** `customerID` (isolated and excluded from model features)
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - **Account & Billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges`

---

## 5. Dataset Cleaning & Validation
1. **Whitespace & Missing Values:** 11 records in raw `TotalCharges` contained whitespace strings (`" "`), all corresponding to new accounts with `tenure == 0` months. These 11 unbilled rows were safely removed, producing 7,032 validated rows.
2. **Type Casting:** `TotalCharges` was converted to `float64`; `SeniorCitizen` was standardized as an integer binary flag.
3. **Partitioning:** Stratified 80/20 train/test split (`train_test_split`, `test_size=0.20`, `random_state=42`):
   - **Training Set:** 5,625 rows (4,130 No, 1,495 Yes — 26.58% churn)
   - **Hold-out Test Set:** 1,407 rows (1,033 No, 374 Yes — 26.58% churn)

---

## 6. Exploratory Data Analysis (EDA)
- **Contract Type:** Month-to-month contract holders exhibited a **42.71%** churn rate, compared to **11.27%** for one-year and **2.83%** for two-year contracts.
- **Tenure:** Churn is front-loaded; median tenure for churners was **10.0 months** vs. **38.0 months** for retained subscribers.
- **Internet Technology & Billing:** Fiber Optic subscribers exhibited higher churn (**41.89%**) relative to DSL (**18.96%**) and No Internet (**7.40%**), driven in part by higher monthly charges ($89.85 median).
- **Payment Method:** Electronic check users exhibited a **45.29%** churn rate vs. automatic payment methods (~15–17%).

---

## 7. Feature Engineering
Two domain-specific features were engineered prior to column transformation:
1. `total_services`: Count of active subscribed services across 8 service-related columns, ranging from 0 to 8.
2. `monthly_charges_diff`: Calculated as `MonthlyCharges - (TotalCharges / max(tenure, 1))`, capturing recent rate changes or expiring promotional pricing.


---

## 8. Preprocessing Pipeline
- **Numerical Pipeline (6 features):** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `total_services`, `monthly_charges_diff` standardized using `StandardScaler()`.
- **Categorical Pipeline (15 features):** Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Output Dimensionality:** Exactly **47 dense numerical features**.
- **Data Leakage Safeguard:** The preprocessor was fitted strictly on `X_train` and serialized to `artifacts/preprocessor.joblib`.

---

## 9. Model Training
Models were trained with fixed random seed `random_state=42` using Scikit-Learn.

---

## 10. Logistic Regression (Primary Model)
- **Configuration:** `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `class_weight=None`.
- **Role:** Primary interpretable model providing continuous probabilistic scores and linear log-odds coefficients.

---

## 11. Random Forest (Benchmark Model)
- **Configuration:** `n_estimators=300`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `criterion='gini'`, `random_state=42`, `n_jobs=-1`.
- **Role:** Non-linear benchmark assessing whether multi-way feature interactions enhance discriminative ranking.

---

## 12. Model Comparison
Both models were evaluated on the identical 1,407-record hold-out test set:

| Metric | Logistic Regression (Selected) | Random Forest (Benchmark) |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.8359** | 0.8175 |
| **Gini Coefficient** | **0.6719** | 0.6351 |
| **KS Statistic** | **0.5066** (@ 0.39) | 0.4913 (@ 0.27) |
| **Brier Score** | **0.1401** | 0.1482 |
| **Accuracy (@ 0.50)** | **80.38%** | 78.61% |
| **Precision (@ 0.50)** | **64.94%** | 62.54% |
| **Recall (@ 0.50)** | **56.95%** | 48.66% |
| **F1-Score (@ 0.50)** | **0.6068** | 0.5474 |

---

## 13. Evaluation Metrics & Interpretability

### Confusion Matrix (@ 0.50 Threshold, Test Set):
- **True Negatives (TN):** 918 | **False Positives (FP):** 115
- **False Negatives (FN):** 161 | **True Positives (TP):** 213

### Top Statistical Churn Drivers:
- **Risk Elevators:** `TotalCharges` ($\beta=+0.6436, \text{OR}=1.90$), `Contract_Month-to-month` ($\beta=+0.6109, \text{OR}=1.84$), `InternetService_Fiber optic` ($\beta=+0.6015, \text{OR}=1.82$).
- **Protective Factors:** `tenure` ($\beta=-1.3547, \text{OR}=0.26$), `Contract_Two year` ($\beta=-0.7680, \text{OR}=0.46$), `InternetService_DSL` ($\beta=-0.6455, \text{OR}=0.52$).

---

## 14. Decision Threshold Analysis

| Threshold | Precision | Recall | F1-Score | Accuracy | Predicted Churners | Business Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.20** | 0.4576 | **0.8663** | 0.5989 | 0.6915 | 708 (50.32%) | High-sensitivity proactive retention |
| **0.30** | 0.5118 | **0.7567** | 0.6106 | 0.7434 | 553 (39.30%) | Balanced retention sweet spot |
| **0.39 (KS)**| 0.5694 | **0.6925** | **0.6248** | 0.7797 | 455 (32.34%) | **Maximum statistical separation (KS = 0.5066)** |
| **0.50** | **0.6494** | 0.5695 | 0.6068 | **0.8038** | 328 (23.31%) | Standard balanced baseline |
| **0.60** | 0.6761 | 0.3850 | 0.4906 | 0.7875 | 213 (15.14%) | High-precision / limited budget |
| **0.70** | **0.7738** | 0.1738 | 0.2838 | 0.7669 | 84 (5.97%) | Targeted concierge intervention |

---

## 15. Operational Churn Risk Bands

| Risk Tier | Probability Range | Test Population | Test Churned | Empirical Churn Rate | Operational Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Low Risk** | $p < 0.20$ | 699 (49.68%) | 50 | **7.15%** | Standard relationship marketing; no discount spend. |
| **Medium Risk** | $0.20 \le p < 0.50$ | 380 (27.01%) | 111 | **29.21%** | Feature adoption prompts and onboarding check-ins. |
| **High Risk** | $0.50 \le p < 0.75$ | 295 (20.97%) | 188 | **63.73%** | Proactive retention discount or contract incentive. |
| **Critical Risk** | $p \ge 0.75$ | 33 (2.35%) | 25 | **75.76%** | Immediate executive outreach and plan remediation. |

---

## 16. FastAPI Model Serving Layer
- `GET /health` — Service readiness probe (`{"status": "healthy"}`).
- `GET /model-info` — Exposes model type, 47 feature dimensions, and risk tier boundaries.
- `POST /predict-churn` — Validates customer payload via Pydantic, applies the saved preprocessor, and returns probability, binary class, and assigned risk tier.

---

## 17. Streamlit Dashboard
An interactive retention dashboard (`src/frontend/app.py`):
- Communicates with FastAPI purely over HTTP/JSON (no direct model loading in UI).
- Includes preset customer archetypes (Low-Risk Loyal, Moderate-Risk Intermediate, High-Risk Attrition).
- Displays color-coded risk dials, predicted probabilities, and risk-tier retention guidance.

---

## 18. Project Architecture

```
[ Raw IBM Telco Dataset ]
            ↓
[ Data Validation & Cleaning (tenure=0 blank handling) ]
            ↓
[ Exploratory Data Analysis & Feature Profiling ]
            ↓
[ Stratified Split (80/20 Train/Test) — Leakage-Safe ]
            ↓
[ Feature Engineering + Scikit-Learn Preprocessing Pipeline ]
            ↓
┌─────────────────────────────────────────────────────────┐
│                       Model Zoo                         │
│  • Logistic Regression (Primary, C=1.0, L2)             │
│  • Random Forest (Benchmark, 300 Trees)                 │
└─────────────────────────────────────────────────────────┘
            ↓
[ Evaluation: ROC-AUC, Gini, KS, Brier & Threshold Analysis ]
            ↓
[ Serialized Artifacts (Joblib, CSV, JSON) ]
            ↓
[ FastAPI REST Microservice (Uvicorn / Pydantic Validation) ]
            ↓ (HTTP / JSON)
[ Streamlit Retention Analytics Dashboard ]
```

---

## 19. Project Structure

```
customer-churn-analytics/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── telco_churn_cleaned.csv
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   ├── validate_data.py
│   │   └── eda.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── model_utils.py
│   │   └── train_models.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── threshold_analysis.py
│   │   └── evaluate_models.py
│   ├── api/
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── main.py
│   └── frontend/
│       └── app.py
├── artifacts/
│   ├── preprocessor.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── logistic_coefficients.csv
│   ├── feature_names.csv
│   ├── model_metadata.json
│   ├── eda/
│   └── evaluation/
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_eda.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   ├── test_frontend.py
│   └── test_integration.py
├── docs/
│   ├── eda_report.md
│   ├── preprocessing_report.md
│   ├── model_training_report.md
│   ├── model_evaluation_report.md
│   ├── api_report.md
│   ├── dashboard_report.md
│   └── final_project_report.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 20. Installation

1. **Clone repository:**
   ```powershell
   git clone https://github.com/Kavyansh180/customer-churn-analytics.git
   cd customer-churn-analytics
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 21. How to Run API

Start the FastAPI backend server:
```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Health Check: `http://127.0.0.1:8000/health`
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- ReDoc Docs: `http://127.0.0.1:8000/redoc`

---

## 22. How to Run Streamlit Dashboard

In a separate terminal window:
```powershell
streamlit run src/frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 23. How to Run Tests

Execute the automated test suite:
```powershell
pytest -q
```
Expected output: `99 passed, 10 warnings`

---

## 24. Example API Request & Response

**Request (`POST http://127.0.0.1:8000/predict-churn`):**
```json
{
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
  "TotalCharges": 29.85
}
```

**Response (`HTTP 200 OK`):**
```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.5847,
  "predicted_churn": 1,
  "risk_band": "High Risk",
  "decision_threshold": 0.5
}
```

---

## 25. Limitations
- **Observational Correlation:** Model coefficients describe historical statistical associations, not causal mechanisms. Retention incentives do not guarantee churn prevention.
- **Uncalibrated Business Value:** The model outputs statistical probabilities on customer churn; it does not estimate customer Lifetime Value (LTV) or intervention ROI.
- **Static Batch Calibration:** Models reflect a single snapshot; production environments require ongoing drift detection and scheduled retraining.
- **Single-Node Prototype:** Designed for modular demonstration and lightweight serving; enterprise deployments require container orchestration and distributed message queues.

---

## 26. Interview-Relevant Project Explanation
- **Why Logistic Regression over Random Forest?** On this 47-feature transformed space, Logistic Regression achieved a higher out-of-sample ROC-AUC (**0.8359** vs. **0.8175**), better probability calibration (Brier score **0.1401** vs. **0.1482**), and direct odds-ratio interpretability for stakeholders.
- **How was Data Leakage Prevented?** Missing value imputation parameters, standard deviation / mean scaling values, and one-hot categorical vocabularies were learned strictly on the 80% training split. The hold-out test set and real-time API payloads are transformed using the persisted pipeline without re-fitting.
- **Why Decoupled Architecture?** The Streamlit UI acts as a stateless HTTP client to the FastAPI microservice. The UI never imports Joblib or Scikit-Learn models, ensuring clear separation of concerns and independent service scalability.
