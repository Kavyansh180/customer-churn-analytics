# Model Training Report
**Project:** Customer Churn Prediction & Retention Analytics System  
**Artifacts Generated:** `artifacts/logistic_regression.joblib`, `artifacts/random_forest.joblib`, `artifacts/logistic_coefficients.csv`, `artifacts/model_metadata.json`  
**Scope:** Phase 5 — Model Training & Interpretability  

---

## 1. Model Selection Rationale

### A. Primary Interpretable Model: Logistic Regression
- **Interpretability:** Maps linear combinations of transformed features to log-odds of customer churn via the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Business Transparency:** Every feature receives an explicit coefficient ($\beta$) and multiplicative Odds Ratio ($\exp(\beta)$), allowing business stakeholders to understand exactly which customer behaviors elevate churn risk.
- **Probabilistic Predictions:** Naturally outputs continuous probability scores $P(\text{Churn} = 1 \mid X) \in [0.0, 1.0]$ suitable for downstream risk tiering (Low, Medium, High).

- **Computational Efficiency:** Fast training and microsecond inference latency, ideal for real-time REST API scoring.

### B. Benchmark Model: Random Forest Classifier
- **Nonlinear & Interaction Capacity:** An ensemble of 300 de-correlated decision trees trained via bootstrap aggregation (bagging) and random feature subspace sampling.
- **Automatic Interactions:** Captures complex multi-way feature interactions (e.g., high `MonthlyCharges` combined with month-to-month contracts and fiber optic service) without requiring manual interaction engineering.
- **Robustness:** Invariant to monotonic feature scales and highly resilient to non-normality.

---

## 2. Training Dataset Specification

- **Training Records (`X_train`):** **5,625 rows**
- **Test Records (`X_test`):** **1,407 rows**
- **Class Balance (Train):** 4,130 retained (`0`) vs. 1,495 churned (`1`) (**26.58% positive class**)
- **Transformed Feature Dimensionality:** **47 dense numerical features** (6 standardized continuous/discrete numerical features + 41 one-hot encoded categorical dummies).

---

## 3. Preprocessing Pipeline Reuse & Data Leakage Prevention

- **Pipeline Reuse:** The fitted Scikit-Learn `Pipeline` serialized in Phase 4 ([artifacts/preprocessor.joblib](file:///d:/customer-churn-analytics/artifacts/preprocessor.joblib)) was loaded directly to transform both training and test sets.
- **Leakage Prevention Safeguard:**
  - **No Test Re-fitting:** Transformer parameters (e.g., standard deviation and mean for `StandardScaler`, category vocabularies for `OneHotEncoder`, median values for `SimpleImputer`) were learned solely from `X_train`.
  - Re-fitting on `X_test` would inject test distribution knowledge into feature transformations, invalidating test set independence and producing over-optimistic generalization claims.

---

## 4. Understanding Model Predictions (`predict_proba`)

In customer retention analytics, binary labels (`0` or `1`) derived from a default 0.5 threshold are insufficient for operational business decisions.

$$\hat{p} = P(\text{Churn} = 1 \mid X) = \text{predict\_proba}(X)[:, 1]$$

- **Why Continuous Probabilities Matter:**
  - A customer with $\hat{p} = 0.52$ has borderline risk, whereas a customer with $\hat{p} = 0.94$ has imminent churn risk.
  - Continuous probabilities allow the business to define custom decision thresholds and segment customers into **Churn Risk Bands** (e.g., Low: $<0.30$, Medium: $0.30–0.60$, High: $>0.60$) to allocate retention budgets effectively.

---

## 5. Logistic Regression Interpretability: Coefficients & Odds Ratios

### Mathematical Foundation

$$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$

$$\text{Odds Ratio (OR)} = e^{\beta_i}$$

- **Coefficient ($\beta_i$):** Represents the change in log-odds of churn for a 1-unit increase in the standardized feature $x_i$, holding all other features constant.
- **Odds Ratio ($e^{\beta_i}$):**
  - $\text{OR} > 1.0$ ($\beta > 0$): Multiplicative increase in churn odds relative to baseline.
  - $\text{OR} < 1.0$ ($\beta < 0$): Multiplicative decrease in churn odds (protective factor).
  - $\text{OR} = 1.0$ ($\beta = 0$): Feature has no linear association with churn log-odds.

### Top Churn-Increasing Factors (Highest Risk)

| Rank | Feature Name | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Direction / Business Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `num__TotalCharges` | **+0.644** | **1.905** | Higher cumulative spend reflects long-term active account billing. |
| **2** | `cat__Contract_Month-to-month` | **+0.611** | **1.842** | Month-to-month contract increases odds of churn by **84.2%** vs. baseline. |
| **3** | `cat__InternetService_Fiber optic` | **+0.602** | **1.825** | Fiber optic subscription increases odds of churn by **82.5%** vs. baseline. |
| **4** | `num__total_services` | **+0.221** | **1.247** | Add-on service count metric. |
| **5** | `cat__OnlineSecurity_No` | **+0.206** | **1.228** | Absence of online security increases churn odds by **22.8%**. |

### Top Churn-Decreasing Factors (Highest Retention)

| Rank | Feature Name | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Direction / Business Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `num__tenure` | **-1.355** | **0.258** | A 1-std increase in tenure decreases odds of churn by **74.2%** ($\text{OR} = 0.258$). |
| **2** | `cat__Contract_Two year` | **-0.768** | **0.464** | Two-year contract commitment reduces churn odds by **53.6%** ($\text{OR} = 0.464$). |
| **3** | `cat__InternetService_DSL` | **-0.646** | **0.524** | DSL service associates with **47.6%** lower churn odds vs. Fiber optic baseline. |
| **4** | `num__MonthlyCharges` | **-0.596** | **0.551** | Controlled monthly charge effect. |
| **5** | `cat__PaperlessBilling_No` | **-0.300** | **0.741** | Traditional paper billing associates with **25.9%** lower churn odds. |

The complete table of all 47 feature coefficients and odds ratios is available at [artifacts/logistic_coefficients.csv](file:///d:/customer-churn-analytics/artifacts/logistic_coefficients.csv).

---

## 6. Critical Methodological Caution: Association vs. Causation

> [!IMPORTANT]
> **Observational Interpretation Warning:**  
> The coefficients and odds ratios reported above describe **statistical associations** observed within historical observational data, **not causal proof**.  
> For instance, while fiber optic service strongly correlates with higher churn odds ($\text{OR} = 1.825$), fiber optic cables do not inherently cause customers to leave. The elevated churn rate may be driven by unmeasured confounding factors, such as local network stability issues, promotional price roll-offs, or competitive offerings from regional ISPs. Interventions (e.g., offering discounts or upgrading hardware) should be validated through A/B testing before enterprise-wide rollout.

---

## 7. Model Training Sanity Checks Summary

| Sanity Verification Test | Status | Observed Value / Result |
| :--- | :---: | :--- |
| **Training Execution** | Passed | Both Logistic Regression & Random Forest converged cleanly without errors |
| **Probability Range Check** | Passed | $P(\text{Churn} = 1) \in [0.0015, 0.8644]$ (LR), $[0.0, 1.0]$ (RF) |
| **Prediction Array Shape** | Passed | Exactly 1,407 predictions matching test split length |
| **NaN / Null / Inf Check** | Passed | 0 NaNs or infinities across all test predictions |
| **Artifact Deserialization** | Passed | Both `.joblib` files reload with matching feature dimensions (47 features) |

---

## 8. Why Evaluation is Deferred to Phase 6

To maintain rigorous separation of concerns in modern data science engineering:
1. **Phase 5** focuses strictly on model fitting, mathematical interpretability, sanity verification, and artifact serialization.
2. **Phase 6** will conduct in-depth, unbiased evaluation on the hold-out test set:
   - Metric computation (Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC)
   - Confusion matrices and error analysis
   - ROC and Precision-Recall curve visualizations
   - Decision threshold tuning and probability calibration assessment
   - Final champion model selection between Logistic Regression and Random Forest
