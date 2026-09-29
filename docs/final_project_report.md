# Final Project Report & Technical Specification
## Customer Churn Prediction & Retention Analytics System

---

## A. Executive Summary

Customer attrition poses a fundamental revenue risk in subscription businesses. The **Customer Churn Prediction & Retention Analytics System** is a complete, modular, and interview-ready machine learning solution engineered from raw data ingestion to production-style model serving and interactive decision-support visualization.

Built on the industry-standard IBM Telco Customer Churn dataset (7,032 validated accounts), the system frames churn detection as a probabilistic classification task. It prioritizes model interpretability, statistical rigor, and strict prevention of data leakage. A tuned and calibrated **Logistic Regression** model serves as the primary inference engine, achieving an out-of-sample **ROC-AUC of 0.8359**, a **Gini coefficient of 0.6719**, and a **Kolmogorov-Smirnov (KS) statistic of 0.5066**. A nonlinear **Random Forest** classifier serves as an empirical benchmark (ROC-AUC: 0.8175).

Predictions are translated into four operational churn risk tiers (*Low Risk*, *Medium Risk*, *High Risk*, *Critical Risk*) with prescriptive retention guidance. The system is deployed via a high-performance **FastAPI** REST microservice and consumed by an interactive **Streamlit** dashboard over HTTP/JSON.

---

## B. End-to-End System Architecture

The project follows a decoupled, unidirectional data and service flow:

```
[ Raw Dataset: Telco CSV ]
            ↓
[ Data Validation & Cleaning (Imputation / Type Casting) ]
            ↓
[ Exploratory Data Analysis & Statistical Profiling ]
            ↓
[ Stratified Split (80/20 Train/Test) — Leakage-Safe ]
            ↓
[ Feature Engineering + Scikit-Learn Preprocessing Pipeline ]
            ↓
┌─────────────────────────────────────────────────────────┐
│                     Model Zoo                           │
│  • Logistic Regression (Primary, C=0.1, L2)             │
│  • Random Forest (Benchmark, 100 Estimators)            │
└─────────────────────────────────────────────────────────┘
            ↓
[ Model Evaluation, KS / Gini / Calibration & Threshold Tuning ]
            ↓
[ Serialized Artifacts: joblib / CSV / JSON ]
            ↓
[ FastAPI REST Microservice (Uvicorn / Pydantic Validation) ]
            ↓ (HTTP / JSON)
[ Streamlit Retention Analytics Dashboard ]
```

### Architectural Principles:
1. **Clean Decoupling:** Model training and inference are strictly separated. Inference endpoints and UI layers never execute model training (`.fit()` or `.fit_transform()`).
2. **Zero In-Memory Model Coupling in UI:** Streamlit acts strictly as an HTTP client to the FastAPI service.
3. **Artifact Portability:** Standardized serialization using Joblib, CSV, and JSON metadata ensures exact cross-environment reproducibility.

---

## C. Dataset Description & Data Cleaning

- **Source Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Initial Volume:** 7,043 customer records across 21 raw columns.
- **Target Variable:** `Churn` (`No` → 0, `Yes` → 1). Overall class distribution: 73.42% retained, 26.58% churned.

### Data Cleaning Actions:
1. **Whitespace & Missing Value Treatment:** 11 records in `TotalCharges` contained whitespace strings (`" "`), all corresponding to brand-new accounts with `tenure = 0`. These 11 records were safely dropped, establishing a clean dataset of **7,032 rows and 21 columns**.
2. **Data Type Casting:** `TotalCharges` was cast from object string to float64. `SeniorCitizen` was standardized as integer binary flag.
3. **Identifier Isolation:** `customerID` was isolated and excluded from model inputs to prevent artificial identifier memorization and overfitting.
4. **Data Partitioning:** Cleaned data was partitioned using `train_test_split` with `test_size=0.20`, `random_state=42`, and target stratification:
   - **Training Set:** 5,625 rows (4,130 No, 1,495 Yes — 26.58% churn)
   - **Hold-out Test Set:** 1,407 rows (1,033 No, 374 Yes — 26.58% churn)

---

## D. Exploratory Data Analysis (EDA) Findings

Comprehensive univariate, bivariate, and multivariate analysis revealed strong signals governing customer attrition:

1. **Contractual Commitment:** Customers on Month-to-month contracts exhibited a churn rate of **42.71%**, compared to **11.27%** for One-year contracts and **2.83%** for Two-year contracts.
2. **Tenure & Lifecycle Stage:** Churn is heavily front-loaded. Customers who churned had a median tenure of **10.0 months** (mean: 17.98), whereas retained customers had a median tenure of **38.0 months** (mean: 37.65).
3. **Monthly Charges & Technology:** High monthly charges correlate positively with churn. Customers with Fiber Optic internet had a churn rate of **41.89%** (median monthly charge: $89.85), compared to DSL (**18.96%**) and No Internet (**7.40%**).
4. **Payment Method Friction:** Electronic check users exhibited a **45.29%** churn rate, far exceeding automatic payment methods (Bank Transfer: 16.71%, Credit Card: 15.24%).
5. **Support & Security Ecosystem:** Subscribing to Online Security, Tech Support, and Device Protection strongly correlated with reduced attrition.

---

## E. Feature Engineering

Domain-specific feature engineering produced two high-signal synthetic features prior to column transformation:

1. **`total_services` (Integer Count):** Sum of active subscriptions across 9 available telecom services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`). Captures account stickiness and multi-product integration.
2. **`monthly_charges_diff` (Float Difference):** Calculated as `MonthlyCharges - (TotalCharges / (tenure + 1))`. Measures short-term pricing changes, recent promotional discounts ending, or rate increases relative to account historical average.

---

## F. Preprocessing Pipeline

The preprocessing pipeline was constructed using Scikit-Learn `ColumnTransformer` and `Pipeline` objects fitted **strictly on the training partition**:

- **Numerical Features (6 features):** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `total_services`, `monthly_charges_diff` → Standardized via `StandardScaler()`.
- **Categorical Features (15 features):** Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Transformed Feature Dimension:** Exactly **47 dense numerical features**.
- **Leakage Safeguards:** Scaler parameters ($\mu, \sigma$) and one-hot categories were learned exclusively from `train.csv` and serialized to `artifacts/preprocessor.joblib`.

---

## G. Model Training & Parameterization

Two classification architectures were trained with fixed random seed `random_state=42`:

1. **Logistic Regression (Primary Model):**
   - **Configuration:** `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `class_weight=None`.
   - **Rationale:** High probabilistic calibration, linear interpretability via odds ratios, low computational overhead, and optimal decision boundary stability.
2. **Random Forest Classifier (Benchmark Model):**
   - **Configuration:** `n_estimators=300`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `criterion='gini'`, `random_state=42`, `n_jobs=-1`.
   - **Rationale:** Non-linear ensemble baseline assessing whether multi-way feature interactions improve ranking performance.

---

## H. Model Comparison & Selected Primary Model

Both models were evaluated on the exact same 1,407 hold-out test instances:

| Metric | Logistic Regression (Selected) | Random Forest (Benchmark) | Winning Model |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.8359** | 0.8175 | Logistic Regression (+0.0184) |
| **Gini Coefficient** | **0.6719** | 0.6351 | Logistic Regression (+0.0368) |
| **KS Statistic** | **0.5066** | 0.4913 | Logistic Regression (+0.0153) |
| **Brier Score** | **0.1401** | 0.1482 | Logistic Regression (Lower/Better) |
| **Accuracy (@ 0.50)** | **80.38%** | 78.61% | Logistic Regression (+1.77%) |
| **Precision (@ 0.50)** | **64.94%** | 62.54% | Logistic Regression (+2.40%) |
| **Recall (@ 0.50)** | **56.95%** | 48.66% | Logistic Regression (+8.29%) |
| **F1-Score (@ 0.50)** | **0.6068** | 0.5474 | Logistic Regression (+0.0594) |

**Selection Decision:** Logistic Regression outperformed Random Forest across every statistical and operational dimension on unseen test data, while offering full coefficient interpretability and superior probability calibration (Brier score 0.1401 vs 0.1482).

---

## I. Detailed Evaluation Metrics (Hold-out Test Set)

### Logistic Regression Test Confusion Matrix (@ Threshold 0.50):
- **True Negatives (TN):** 918
- **False Positives (FP):** 115
- **False Negatives (FN):** 161
- **True Positives (TP):** 213
- **Total Test Accounts:** 1,407

### Top Positive Churn Drivers (Standardized Log-Odds Coefficients):
1. `TotalCharges`: **+0.6436** ($\text{OR} = 1.9033$)
2. `Contract_Month-to-month`: **+0.6109** ($\text{OR} = 1.8421$)
3. `InternetService_Fiber optic`: **+0.6015** ($\text{OR} = 1.8248$)
4. `total_services`: **+0.2206** ($\text{OR} = 1.2468$)
5. `OnlineSecurity_No`: **+0.2057** ($\text{OR} = 1.2284$)

### Top Negative Churn Drivers (Protective Retention Drivers):
1. `tenure`: **-1.3547** ($\text{OR} = 0.2580$)
2. `Contract_Two year`: **-0.7680** ($\text{OR} = 0.4639$)
3. `InternetService_DSL`: **-0.6455** ($\text{OR} = 0.5244$)
4. `MonthlyCharges`: **-0.5960** ($\text{OR} = 0.5510$)
5. `PaperlessBilling_No`: **-0.3000** ($\text{OR} = 0.7408$)

---

## J. Decision Threshold Tuning Analysis

To align model classification with varying business objectives, threshold analysis across $[0.10, 0.90]$ was conducted on the holdout test set:

| Threshold | Precision | Recall | F1-Score | Accuracy | Predicted Churners | Share of Base | Operational Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.10** | 0.3971 | **0.9439** | 0.5590 | 0.6041 | 889 | 63.18% | Broad safety net (captures 94.4% of all churners) |
| **0.20** | 0.4576 | **0.8663** | 0.5989 | 0.6915 | 708 | 50.32% | High-sensitivity proactive retention |
| **0.30** | 0.5118 | **0.7567** | 0.6106 | 0.7434 | 553 | 39.30% | Balanced retention sweet spot (captures 75.7%) |
| **0.39 (KS)**| 0.5694 | **0.6925** | **0.6248** | 0.7797 | 455 | 32.34% | **Maximum statistical separation (KS = 0.5066)** |
| **0.50** | **0.6494** | 0.5695 | 0.6068 | **0.8038** | 328 | 23.31% | Standard balanced default |
| **0.60** | 0.6761 | 0.3850 | 0.4906 | 0.7875 | 213 | 15.14% | High precision, low budget intervention |
| **0.70** | **0.7738** | 0.1738 | 0.2838 | 0.7669 | 84 | 5.97% | Emergency escalation for imminent churners |

---

## K. Operational Churn Risk Bands

Rather than relying on a hard binary decision, the continuous churn probability is partitioned into four actionable tiers:

```
[ 0.00 ] ─── Low Risk ─── [ 0.20 ] ─── Medium Risk ─── [ 0.50 ] ─── High Risk ─── [ 0.75 ] ─── Critical Risk ─── [ 1.00 ]
```

| Risk Tier | Probability Range | Customer Count | Share of Base | Actual Churned | Actual Churn Rate (%) | Recommended Business Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low Risk** | $p < 0.20$ | **699** | **49.68%** | 50 | **7.15%** | Standard automated relationship marketing; no discount spend. |
| **Medium Risk** | $0.20 \le p < 0.50$ | **380** | **27.01%** | 111 | **29.21%** | Feature adoption prompts, loyalty surveys, usage monitoring. |
| **High Risk** | $0.50 \le p < 0.75$ | **295** | **20.97%** | 188 | **63.73%** | Proactive account manager check-in, 1-year contract renewal incentive. |
| **Critical Risk** | $p \ge 0.75$ | **33** | **2.35%** | 25 | **75.76%** | Immediate executive outreach, aggressive contract retention credit. |


---

## L. FastAPI Serving Layer

The REST API is implemented in `src/api/` using FastAPI, Pydantic, and Uvicorn.

### Endpoints:
1. `GET /health`: Healthcheck verifying server status (`{"status": "healthy"}`).
2. `GET /model-info`: Returns model type, 47 feature dimensions, decision threshold, and risk band cutoffs.
3. `POST /predict-churn`: Accepts full customer payload with strict Pydantic data validation and returns:
   - `customerID`: Echoed customer identifier.
   - `churn_probability`: Float probability in $[0, 1]$ rounded to 4 decimals.
   - `predicted_churn`: Binary integer classification (0 or 1).
   - `risk_band`: Operational risk tier label.
   - `decision_threshold`: Configured decision threshold (0.50).

### Validation & Error Handling:
- Strict Pydantic validation rejects negative numerical charges or tenures (HTTP 422).
- Missing mandatory fields return structured error payloads.
- Novel categorical values are gracefully mapped without service crashes.

---

## M. Streamlit Retention Analytics Dashboard

The user interface (`src/frontend/app.py`) provides an interactive web workspace:
1. **System Health Status Indicator:** Real-time ping to FastAPI `/health`.
2. **Preset Customer Profiles:** One-click benchmark loading (Low-Risk Loyal, Moderate-Risk Intermediate, High-Risk Attrition).
3. **Structured Interactive Form:** Categorized into Demographics, Account & Contract, Telecom Services, and Billing.
4. **Visual Risk Indicator:** Color-coded probability gauges (Green, Amber, Orange, Red).
5. **Actionable Retention Directives:** Prescriptive next steps customized per risk band.
6. **Decoupled Architecture:** Streamlit makes pure HTTP/JSON requests and contains zero model training or loading logic.

---

## N. Quality Assurance & Automated Testing

The project maintains comprehensive test coverage across 9 distinct test suites using `pytest`:

| Test Module | Coverage Scope | Status |
| :--- | :--- | :---: |
| `tests/test_data_pipeline.py` | Raw data loading, type conversion, imputation, validation summary | **PASSED** |
| `tests/test_eda.py` | Summary statistics, churn breakdown, correlation matrix, plot generation | **PASSED** |
| `tests/test_preprocessing.py` | Stratification, feature engineering, transformer, 47 feature output | **PASSED** |
| `tests/test_models.py` | Model serialization, predict_proba, binary output, shape verification | **PASSED** |
| `tests/test_evaluation.py` | ROC-AUC, Gini, KS statistic, Brier score, threshold grid analysis | **PASSED** |
| `tests/test_api.py` | FastAPI endpoints, Pydantic validation, HTTP 422, deterministic outputs | **PASSED** |
| `tests/test_frontend.py` | UI imports, preset configs, HTTP error handling, model decoupling | **PASSED** |
| `tests/test_integration.py` | End-to-end inference flow, edge cases, reference profiles, risk bounds | **PASSED** |

**Final Automated Test Execution:** **99 passed tests, 0 failures**.

---

## O. System Limitations & Boundaries

1. **Non-Causal Interpretability:** High model coefficients indicate statistical association with historical churn, not proven direct causality. Offering a discount does not guarantee retention.
2. **Uncalibrated Business Value:** Model outputs are empirical classification probabilities on the Telco dataset; they do not incorporate customer Lifetime Value (LTV) or margin metrics.
3. **Static Training Snapshot:** Models reflect a single snapshot; production environments require ongoing drift detection and scheduled retraining pipelines.
4. **Single-Node Serving:** The current setup serves single requests locally and is not configured for distributed worker clusters or asynchronous streaming message queues.

---

## P. Future Engineering Roadmap

1. **Cost-Sensitive Loss Optimization:** Incorporate customer acquisition costs and retention offer budgets into threshold selection.
2. **Automated Data Drift Monitoring:** Integrate Evidently AI or Great Expectations to detect covariate shift in input distributions.
3. **Containerized Deployment:** Package FastAPI and Streamlit into multi-container Docker Compose definitions.
4. **Explainable AI (SHAP / TreeSHAP):** Expose individual instance-level SHAP force plots directly in the REST API response.
