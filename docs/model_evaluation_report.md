# Model Evaluation, Threshold Analysis & Churn Risk Banding Report
**Project:** Customer Churn Prediction & Retention Analytics System  
**Hold-out Test Dataset:** `data/processed/test.csv` (1,407 records, 26.58% empirical churn rate)  
**Artifacts Generated:** `artifacts/evaluation/model_comparison.csv`, `artifacts/evaluation/threshold_analysis.csv`, `artifacts/evaluation/test_predictions.csv`, `artifacts/evaluation/*.png`  
**Scope:** Phase 6 — Comprehensive Hold-Out Test Evaluation  

---

## 1. Executive Summary & Model Performance Comparison

Both the primary interpretable model (**Logistic Regression**) and the nonlinear ensemble (**Random Forest**) were evaluated on the untouched 1,407-record test set using the fitted Phase 4 preprocessing pipeline.

### Model Performance Matrix (Hold-out Test Set)

| Metric | Logistic Regression (Primary) | Random Forest (Benchmark) | Metric Definition & Objective |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.8359** | **0.8175** | Discriminative ranking power across all thresholds (1.0 = perfect) |
| **Gini Coefficient** | **0.6719** | **0.6351** | Normalized discriminatory index ($2 \times \text{AUC} - 1$) |
| **KS Statistic** | **0.5066** | **0.4913** | Maximum separation between cumulative churner and non-churner CDFs |
| **KS Optimal Threshold** | **0.3900** | **0.2733** | Probability threshold maximizing distribution divergence |
| **Brier Score Loss** | **0.1401** | **0.1482** | Probability calibration & accuracy error (lower is better, 0.0 = perfect) |
| **Accuracy (@ 0.50)** | **0.8038** | **0.7861** | Overall fraction of correct predictions |
| **Precision (@ 0.50)** | **0.6494** | **0.6254** | True Positives / Total Predicted Positive |
| **Recall (@ 0.50)** | **0.5695** | **0.4866** | True Positives / Total Actual Positive Churners |
| **F1-Score (@ 0.50)** | **0.6068** | **0.5474** | Harmonic mean of precision and recall |

*Visual Reference:* `artifacts/evaluation/roc_curves.png`, `artifacts/evaluation/precision_recall_curves.png`

---

## 2. Why Accuracy Alone is Insufficient for Churn Prediction

The test set contains **1,033 retained customers (73.42%)** and **374 churned customers (26.58%)**.
- A naive dummy model that unconditionally predicts every customer will stay achieves **73.42% accuracy**.
- However, that model has **0.0% recall**, completely failing to detect any of the 374 leaving customers, leading to unmitigated revenue leakage.
- Evaluating churn systems requires threshold-independent ranking metrics (**ROC-AUC, KS, Gini**) combined with targeted trade-off metrics (**Precision, Recall, F1, and Cost-Weighted Confusion Matrices**).

---

## 3. Confusion Matrix Breakdown & Business Error Analysis

### Actual Confusion Matrix Values (@ 0.50 Decision Threshold)

| Outcome Category | Logistic Regression Count | Random Forest Count | Business Definition & Financial Impact |
| :--- | :---: | :---: | :--- |
| **True Negative (TN)** | **918** | **924** | Customer predicted to stay and actually stayed. No unnecessary retention spend. |
| **False Positive (FP)** | **115** | **109** | Customer predicted to churn who actually stayed. Incurs cost of unneeded retention discount/call. |
| **False Negative (FN)** | **161** | **192** | Customer predicted to stay who actually churned. **Severe loss of customer LTV & recurring revenue.** |
| **True Positive (TP)** | **213** | **182** | Customer predicted to churn and actually churned. Successfully targeted for retention intervention. |

*Visual References:*  
- `artifacts/evaluation/logistic_confusion_matrix.png`  
- `artifacts/evaluation/random_forest_confusion_matrix.png`

### Asymmetric Cost of Errors:
- **False Negative Cost ($C_{\text{FN}}$):** Forfeited customer lifetime value (e.g., $1,000–$2,500 cumulative revenue) plus higher future acquisition cost.
- **False Positive Cost ($C_{\text{FP}}$):** Cost of proactive outreach, special discount codes, or promotional gift vouchers (e.g., $15–$50).
- Because $C_{\text{FN}} \gg C_{\text{FP}}$, the default 0.50 threshold is rarely optimal for customer retention.

---

## 4. Decision Threshold Analysis (Logistic Regression)

Evaluating thresholds from **0.10 to 0.90** reveals the direct operational trade-off between precision (intervention efficiency) and recall (churn capture rate).

| Threshold | Precision | Recall | F1-Score | Accuracy | Predicted Churners | Predicted Churn Share | Operational Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.10** | 0.3971 | **0.9439** | 0.5590 | 0.6041 | 889 | 63.18% | Broad safety net (captures 94.4% of all churners) |
| **0.20** | 0.4576 | **0.8663** | 0.5989 | 0.6915 | 708 | 50.32% | High-sensitivity proactive retention |
| **0.30** | 0.5118 | **0.7567** | 0.6106 | 0.7434 | 553 | 39.30% | Balanced retention sweet spot (captures 75.7%) |
| **0.39 (KS)**| 0.5694 | **0.6925** | **0.6248** | 0.7797 | 455 | 32.34% | **Maximum statistical separation (KS = 0.5066)** |
| **0.50** | **0.6494** | 0.5695 | 0.6068 | **0.8038** | 328 | 23.31% | Standard balanced default |
| **0.60** | 0.6761 | 0.3850 | 0.4906 | 0.7875 | 213 | 15.14% | High precision, low budget intervention |
| **0.70** | **0.7738** | 0.1738 | 0.2838 | 0.7669 | 84 | 5.97% | Emergency escalation for imminent churners |

*Visual Reference:* `artifacts/evaluation/threshold_analysis.csv`

---

## 5. Operational Churn Risk Banding System

To translate continuous probabilities $\hat{p} = P(\text{Churn} = 1 \mid X)$ into tiered business workflows, accounts are classified into **4 Operational Churn Risk Bands**:

| Risk Tier | Probability Range | Customer Count | Share of Base | Actual Churned | Actual Churn Rate (%) | Recommended Business Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Low Risk** | $p < 0.20$ | **699** | **49.68%** | 50 | **7.15%** | Standard automated relationship marketing; no discount spend. |
| **Medium Risk** | $0.20 \le p < 0.50$ | **380** | **27.01%** | 111 | **29.21%** | Feature adoption prompts, loyalty surveys, usage monitoring. |
| **High Risk** | $0.50 \le p < 0.75$ | **295** | **20.97%** | 188 | **63.73%** | Proactive account manager check-in, 1-year contract renewal incentive. |
| **Critical Risk** | $p \ge 0.75$ | **33** | **2.35%** | 25 | **75.76%** | Immediate executive outreach, aggressive contract retention credit. |

*Audit File:* `artifacts/evaluation/test_predictions.csv` (1,407 individual scored customer accounts with assigned risk tiers).

---

## 6. Probability Calibration Assessment & Reliability Diagram

- **Brier Score Loss:** **0.1401** (Logistic Regression), confirming sharp, well-calibrated probabilities.
- **Reliability Diagram Inspection:** In `artifacts/evaluation/logistic_calibration_curve.png`, empirical positive fractions closely track the 45-degree ideal calibration line across all 10 uniform probability bins.
- **Operational Interpretation:** When Logistic Regression assigns a 60% probability to an account segment, approximately 60% of those customers actually churned historically.

---

## 7. Association vs. Causation & Operational Tiering Notes

> [!IMPORTANT]
> **Methodological Reminders for Business Teams:**
> 1. **Statistical Association $\neq$ Causal Guarantee:** Model risk factors (e.g., electronic check payment method having $\text{OR} = 1.84$) reflect historical correlation patterns, not direct mechanical causes.
> 2. **Operational Bands $\neq$ Rigid Laws:** The 4 risk tiers ($<0.20, 0.20–0.50, 0.50–0.75, \ge 0.75$) represent pragmatic, operational segments to structure retention workflows. Optimal production boundaries should adjust dynamically based on retention team staffing capacity and unit economics.

---

*Report generated during Phase 6 execution.*
