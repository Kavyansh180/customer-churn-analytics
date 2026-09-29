# Exploratory Data Analysis (EDA) Report
**Project:** Customer Churn Prediction & Retention Analytics System  
**Dataset:** `data/processed/telco_churn_cleaned.csv`  
**Target Variable:** `Churn` (`"Yes"` / `"No"`)  
**Scope:** Phase 3 — Baseline Exploratory Analysis  

---

## 1. Executive Summary & Dataset Overview

This exploratory analysis evaluates 7,032 validated customer records to identify key drivers of customer attrition. The primary goal is to surface actionable data patterns across customer demographics, subscribed services, contract structures, and billing practices to guide feature engineering and predictive modeling.

### Dataset Profile

| Attribute | Value | Description |
| :--- | :--- | :--- |
| **Total Rows** | **7,032** | Cleaned customer accounts with valid billing history |
| **Total Columns** | **21** | 1 target, 1 ID, 3 continuous numeric, 1 binary numeric, 15 categorical |
| **Target Variable** | `Churn` | Binary indicator of customer departure |
| **Numerical Features** | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` | Continuous and discrete financial/tenure metrics |
| **Categorical Features** | 15 features | Demographics, service options, contract terms, payment methods |
| **Missing Values** | **0** | Cleaned in Phase 2 |

---

## 2. Target Variable (`Churn`) Distribution

The dataset exhibits moderate class imbalance with an approximate 2.76 : 1 ratio of retained to churned customers.

| Class Label | Customer Count | Proportion (%) | Business Implication |
| :--- | :--- | :--- | :--- |
| **`No` (Retained)** | 5,163 | **73.42%** | Majority base generating steady recurring revenue |
| **`Yes` (Churned)** | 1,869 | **26.58%** | ~1 in every 4 customers churns; significant revenue leakage |
| **Total** | **7,032** | **100.00%** | Baseline accuracy benchmark = 73.42% |

*Visual Reference:* `artifacts/eda/01_target_distribution.png`

---

## 3. Numerical Features Univariate & Bivariate Findings

### A. Overall Statistical Distribution

| Feature | Mean | Median | Std Dev | Min | Q25 (25%) | Q75 (75%) | Max | IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tenure` (Months)** | 32.42 | 29.00 | 24.55 | 1.00 | 9.00 | 55.00 | 72.00 | 46.00 |
| **`MonthlyCharges` ($)** | $64.80 | $70.35 | $30.09 | $18.25 | $35.59 | $89.86 | $118.75 | $54.28 |
| **`TotalCharges` ($)** | $2,283.30 | $1,397.48 | $2,266.77 | $18.80 | $401.45 | $3,794.74 | $8,684.80 | $3,393.29 |

### B. Group-Level Comparison: Retained (`No`) vs. Churned (`Yes`)

| Feature | Churn Status | Mean | Median | Std Dev | IQR | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tenure` (Months)** | **Retained (`No`)** | **37.65** | **38.00** | 24.08 | 46.00 | 1.00 | 72.00 |
| | **Churned (`Yes`)** | **17.98** | **10.00** | 19.53 | 27.00 | 1.00 | 72.00 |
| **`MonthlyCharges` ($)** | **Retained (`No`)** | **$61.31** | **$64.45** | $31.09 | $63.38 | $18.25 | $118.75 |
| | **Churned (`Yes`)** | **$74.44** | **$79.65** | $24.67 | $38.05 | $18.85 | $118.35 |
| **`TotalCharges` ($)** | **Retained (`No`)** | **$2,555.34** | **$1,683.60** | $2,329.46 | $3,686.30 | $18.80 | $8,672.45 |
| | **Churned (`Yes`)** | **$1,531.80** | **$703.55** | $1,890.82 | $2,196.80 | $18.85 | $8,684.80 |

### Key Numerical Observations:
1. **Tenure as an Early Warning Indicator:** The median tenure of churned customers is only **10 months**, compared to **38 months** for retained customers. The highest volume of customer departures occurs within the first 1–12 months.
2. **Monthly Price Sensitivity:** Churned customers have substantially higher median monthly charges (**$79.65** vs. **$64.45**).
3. **Total Spend Paradox:** TotalCharges is lower for churned accounts ($1,531.80 vs. $2,555.34) strictly because their account lifetime (tenure) was cut short before cumulative revenue could accrue.

*Visual References:*  
- `artifacts/eda/02_univariate_numerical_distributions.png`  
- `artifacts/eda/03_churn_vs_numerical_boxplots.png`

---

## 4. Categorical Features & Churn Rate Analysis

### A. Contract Terms & Account Setup

| Feature | Category | Total Customers | Churn Count | Churn Rate (%) | Share of Base (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Contract`** | **Month-to-month** | 3,875 | 1,655 | **42.71%** | 55.11% |
| | **One year** | 1,472 | 166 | **11.28%** | 20.93% |
| | **Two year** | 1,685 | 48 | **2.85%** | 23.96% |
| **`PaperlessBilling`** | **Yes** | 4,168 | 1,400 | **33.59%** | 59.27% |
| | **No** | 2,864 | 469 | **16.38%** | 40.73% |
| **`PaymentMethod`** | **Electronic check** | 2,365 | 1,071 | **45.29%** | 33.63% |
| | **Mailed check** | 1,604 | 308 | **19.20%** | 22.81% |
| | **Bank transfer (auto)** | 1,542 | 258 | **16.73%** | 21.93% |
| | **Credit card (auto)** | 1,521 | 232 | **15.25%** | 21.63% |

*Visual Reference:* `artifacts/eda/04_churn_by_key_account_features.png`

### B. Internet Infrastructure & Support Services

| Service Feature | Category | Total Customers | Churn Count | Churn Rate (%) | Share of Base (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`InternetService`** | **Fiber optic** | 3,096 | 1,297 | **41.89%** | 44.03% |
| | **DSL** | 2,416 | 459 | **18.99%** | 34.36% |
| | **No internet** | 1,520 | 113 | **7.43%** | 21.62% |
| **`TechSupport`** | **No** | 3,472 | 1,446 | **41.65%** | 49.37% |
| | **Yes** | 2,040 | 310 | **15.20%** | 29.01% |
| | **No internet** | 1,520 | 113 | **7.43%** | 21.62% |
| **`OnlineSecurity`** | **No** | 3,497 | 1,461 | **41.78%** | 49.73% |
| | **Yes** | 2,015 | 295 | **14.64%** | 28.65% |
| | **No internet** | 1,520 | 113 | **7.43%** | 21.62% |
| **`OnlineBackup`** | **No** | 3,087 | 1,233 | **39.94%** | 43.90% |
| | **Yes** | 2,425 | 523 | **21.57%** | 34.49% |
| **`DeviceProtection`** | **No** | 3,094 | 1,211 | **39.14%** | 44.00% |
| | **Yes** | 2,418 | 545 | **22.54%** | 34.39% |

*Visual Reference:* `artifacts/eda/05_services_churn_rates.png`

### C. Demographics & Household Structure

| Demographic | Category | Total Customers | Churn Count | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **`SeniorCitizen`** | **Yes (1)** | 1,142 | 476 | **41.68%** |
| | **No (0)** | 5,890 | 1,393 | **23.65%** |
| **`Partner`** | **No** | 3,639 | 1,200 | **32.98%** |
| | **Yes** | 3,393 | 669 | **19.72%** |
| **`Dependents`** | **No** | 4,933 | 1,543 | **31.28%** |
| | **Yes** | 2,099 | 326 | **15.53%** |
| **`gender`** | **Female** | 3,483 | 939 | **26.96%** |
| | **Male** | 3,549 | 930 | **26.20%** |

---

## 5. Numerical Correlation Analysis

| Feature | `SeniorCitizen` | `tenure` | `MonthlyCharges` | `TotalCharges` | `Churn_Binary` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SeniorCitizen`** | 1.00 | 0.02 | 0.22 | 0.10 | **+0.15** |
| **`tenure`** | 0.02 | 1.00 | 0.25 | 0.83 | **-0.35** |
| **`MonthlyCharges`** | 0.22 | 0.25 | 1.00 | 0.65 | **+0.19** |
| **`TotalCharges`** | 0.10 | 0.83 | 0.65 | 1.00 | **-0.20** |
| **`Churn_Binary`** | **+0.15** | **-0.35** | **+0.19** | **-0.20** | 1.00 |

*Visual Reference:* `artifacts/eda/06_correlation_heatmap.png`

> [!IMPORTANT]
> **Correlation vs. Causation Notice:** Linear Pearson correlation measures pairwise co-variation, not causal mechanisms. For example, while `tenure` has a moderate negative linear correlation with churn ($r = -0.35$), customer tenure is also confounded by customer satisfaction, onboarding success, and contract commitment.

---

## 6. Outlier Analysis & Data Integrity

The 1.5 × IQR standard outlier rule was applied across all continuous numerical attributes:

| Feature | Q25 | Q75 | IQR | Lower Fence | Upper Fence | Outlier Count | Action / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tenure`** | 9.00 | 55.00 | 46.00 | -60.00 | 124.00 | **0 (0.0%)** | Retain all values [1, 72] |
| **`MonthlyCharges`** | $35.59 | $89.86 | $54.28 | -$45.82 | $171.27 | **0 (0.0%)** | Retain all values [$18.25, $118.75] |
| **`TotalCharges`** | $401.45 | $3,794.74 | $3,393.29 | -$4,688.48 | $8,884.67 | **0 (0.0%)** | Retain all values [$18.80, $8,684.80] |

**Assessment:** All numerical entries reflect legitimate, bounded business values consistent with telecommunication plan pricing and subscription durations. No trimming or synthetic clipping is required.

---

## 7. Business Insights: Observed Associations vs. Potential Business Explanations

| # | Topic | Observed Association (Data Fact) | Potential Business Explanation (Hypothesis) | Actionable Retention Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Contract Length** | Month-to-month contracts have **42.71% churn** vs. **2.85%** for two-year contracts. | Low switching friction allows month-to-month customers to leave whenever dissatisfaction arises. | Offer discount incentives to transition month-to-month users into 1- or 2-year commitments. |
| **2** | **Early Tenure Attrition** | Median tenure of churned customers is **10 months** (vs. 38 for retained). | Inadequate onboarding experience, initial service friction, or unmet promotional expectations. | Deploy high-touch onboarding and 30/60/90-day retention check-ins. |
| **3** | **Fiber Optic Attrition** | Fiber optic churn is **41.89%** vs. DSL at **18.99%** and No Internet at **7.43%**. | Higher price point ($70–$100/mo) coupled with service outages or mismatched price-to-value perception. | Audit fiber network reliability and review pricing competitiveness. |
| **4** | **Payment Friction** | Electronic check users churn at **45.29%** vs. auto-pay methods at **~15.5%**. | Manual monthly payment creates recurring price awareness and payment friction. | Offer a recurring bill credit ($5/mo) for adopting automated bank/credit card payment. |
| **5** | **Protective Tech Services** | Customers without `TechSupport` or `OnlineSecurity` churn at **~41.7%** vs. **~15%** for active subscribers. | Value-added services increase product stickiness and dependency on the provider ecosystem. | Bundle complimentary security and tech support packages into high-risk fiber plans. |
| **6** | **Demographic Stability** | Single customers without dependents churn at **~32%** vs. **~16%** with dependents. | Multi-user family accounts have higher switching friction (multiple SIMs, shared Wi-Fi). | Promote family multi-line bundle plans to anchor households. |

---

## 8. Limitations of the EDA

1. **Observational & Cross-Sectional:** The dataset represents a historical snapshot; it lacks time-series event logs (e.g., historical ticket escalations, dropped calls, speed degradations).
2. **Unmeasured Customer Sentiment:** The data contains no direct CSAT (Customer Satisfaction) or NPS (Net Promoter Score) metrics.
3. **Absence of Competitor Context:** Local geographic competitor pricing and regional network coverage availability are unobserved variables that may influence fiber optic churn.

---

*Report generated as part of Phase 3 execution.*
