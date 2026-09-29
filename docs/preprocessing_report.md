# Preprocessing and Feature Engineering Report
**Project:** Customer Churn Prediction & Retention Analytics System  
**Dataset:** `data/processed/telco_churn_cleaned.csv`  
**Artifacts Generated:** `artifacts/preprocessor.joblib`, `artifacts/feature_names.csv`, `data/processed/train.csv`, `data/processed/test.csv`  
**Scope:** Phase 4 — Data Splitting, Preprocessing Pipeline & Feature Engineering  

---

## 1. Feature & Target Formulation ($X$ and $y$)

- **Target Variable ($y$):** `Churn` converted to binary integers:
  - `"No"` $\rightarrow$ `0` (Retained, negative class)
  - `"Yes"` $\rightarrow$ `1` (Churned, positive class)
- **Predictor Feature Matrix ($X$):** 19 raw feature columns (4 numerical, 15 categorical).

### Why `customerID` Is Excluded from Predictors:
1. **Arbitrary High-Cardinality Identifier:** `customerID` is a synthetic, randomly assigned customer key (`7043` unique alphanumeric values).
2. **Overfitting & Spurious Correlation:** Including arbitrary IDs would allow tree-based algorithms to partition on specific customer strings or linear models to assign spurious coefficients, destroying generalizability.
3. **Data Leakage Risk:** Customer IDs carry zero causal or behavioral signal for future, unseen customers.

---

## 2. Train / Test Split & Stratification Strategy

The dataset is partitioned into training and validation sets using Scikit-Learn's `train_test_split`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

### Partitioning Summary

| Split | Number of Rows | Share of Data | Retained (`0`) | Churned (`1`) | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Dataset** | 7,032 | 100.0% | 5,163 | 1,869 | **26.58%** |
| **Train Set (`X_train`)** | **5,625** | **80.0%** | **4,130** | **1,495** | **26.58%** |
| **Test Set (`X_test`)** | **1,407** | **20.0%** | **1,033** | **374** | **26.58%** |

### Why Stratification is Critical:
Because customer churn is moderately imbalanced (~26.58% positive class), random unstratified sampling risks assigning disproportionate churn rates to the test split (e.g., 22% in train vs. 31% in test). Stratifying on $y$ guarantees that both subsets mirror the empirical class prior exactly down to decimal precision, eliminating split-induced evaluation bias.

---

## 3. Data Leakage Prevention Architecture

Data leakage occurs when information from outside the training dataset is inadvertently used to fit feature transformers, leading to overly optimistic cross-validation and degraded production inference.

### How the Pipeline Enforces Strict Leakage Prevention:
1. **Isolated Fit (`fit(X_train)`):** Preprocessing statistics (means, standard deviations, imputation medians, mode frequencies, and One-Hot category vocabularies) are calculated **strictly from `X_train`**.
2. **Immutable Test Transformation (`transform(X_test)`):** The test set is transformed purely as an unseen input stream without updating any transformer parameters.
3. **Single Composite Pipeline Object:** Feature engineering, imputation, scaling, and one-hot encoding are bundled into a single Scikit-Learn `Pipeline`. When saved to `artifacts/preprocessor.joblib`, the artifact contains all learned parameters, eliminating any manual preprocessing steps at serving time.

```
Raw Input Data (X_train) 
   ──> [ FeatureEngineer ] 
   ──> [ ColumnTransformer: SimpleImputer + StandardScaler / OneHotEncoder ] 
   ──> Preprocessed Array (5,625 × 47)
```

---

## 4. Feature Engineering Decisions

Two domain-justified, leakage-safe features were engineered:

### A. `total_services` (Integer Count: 0 to 8)
- **Formula:** $\sum_{i=1}^{8} \mathbb{I}(\text{Service}_i == \text{"Yes"})$ across `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Business Rationale:** Quantifies customer ecosystem lock-in and product adoption depth.
- **Empirical Justification:** In Phase 3 EDA, customers with 0–1 services churned at **~35–44%**, whereas customers with 7–8 bundled services churned at only **~5–12%**.

### B. `monthly_charges_diff` (Continuous Float, $)
- **Formula:** $\text{MonthlyCharges} - \frac{\text{TotalCharges}}{\max(\text{tenure}, 1.0)}$
- **Business Rationale:** Compares the current monthly recurring charge against the customer's historical average monthly spend over their entire lifetime.
- **Interpretation:**
  - $\text{Positive } (>0):$ Current bill is higher than historical average, indicating price increases or expiration of introductory promotional discounts (a known catalyst for churn).
  - $\text{Negative } (<0):$ Current bill is lower than historical average, reflecting service downgrades or added loyalty discounts.
  - $\text{Zero } (\approx 0):$ Consistent, stable billing history.
- **Zero-Tenure Safety:** Protected with `np.maximum(tenure, 1.0)` to guarantee zero-division safety.

---

## 5. Scikit-Learn Preprocessing Pipeline Details

### A. Numerical Sub-Pipeline (`6 Features`)
- **Features:** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `total_services`, `monthly_charges_diff`.
- **Steps:**
  1. `SimpleImputer(strategy="median")`: Robust to potential extreme values.
  2. `StandardScaler()`: Scales each numerical feature to $\mu = 0, \sigma = 1$, ensuring regularized Logistic Regression and gradient-based estimators converge stably.

### B. Categorical Sub-Pipeline (`15 Features`)
- **Features:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
- **Steps:**
  1. `SimpleImputer(strategy="most_frequent")`: Handles any potential missing categorical entries.
  2. `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`:
     - Converts categorical levels into dense binary indicator columns.
     - `handle_unknown="ignore"` ensures novel categories encountered in production (e.g., a newly introduced payment method) are safely encoded as all zeros rather than raising runtime exceptions.

---

## 6. Transformed Feature Dimensions

| Stage | Row Count | Feature Count | Description |
| :--- | :--- | :--- | :--- |
| **Raw Predictors ($X$)** | 5,625 / 1,407 | 19 | 4 numerical + 15 categorical (excluding `customerID`, `Churn`) |
| **After Feature Engineering** | 5,625 / 1,407 | 21 | + `total_services`, + `monthly_charges_diff` |
| **After One-Hot Encoding & Scaling** | **5,625 (train)** / **1,407 (test)** | **47** | **6 scaled numericals + 41 one-hot dummy features** |

The complete list of all 47 transformed feature names is versioned at [artifacts/feature_names.csv](file:///d:/customer-churn-analytics/artifacts/feature_names.csv).

---

## 7. Persisted Artifacts

1. **`artifacts/preprocessor.joblib`:** Serialized Scikit-Learn `Pipeline` ready for model training in Phase 5 and API inference in Phase 6.
2. **`artifacts/feature_names.csv`:** Auditable mapping of transformed column indices, feature names, and groups.
3. **`data/processed/train.csv`:** Stratified training dataset (5,625 rows).
4. **`data/processed/test.csv`:** Stratified testing dataset (1,407 rows).
