"""Threshold analysis and operational churn risk banding module."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DEFAULT_THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
]


def evaluate_threshold_grid(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Evaluate classification metrics across a spectrum of decision thresholds.

    Args:
        y_true: Binary ground truth array.
        y_prob: Predicted probability array.
        thresholds: List of probability thresholds to evaluate.

    Returns:
        pd.DataFrame summarizing precision, recall, F1, and predicted churn volume.
    """
    th_list = thresholds or DEFAULT_THRESHOLDS
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    total_samples = len(y_true_arr)

    rows = []
    for th in th_list:
        preds = (y_prob_arr >= th).astype(int)
        churn_count = int(preds.sum())
        churn_pct = (churn_count / total_samples) * 100 if total_samples > 0 else 0.0

        p = float(precision_score(y_true_arr, preds, zero_division=0))
        r = float(recall_score(y_true_arr, preds, zero_division=0))
        f = float(f1_score(y_true_arr, preds, zero_division=0))
        acc = float(accuracy_score(y_true_arr, preds))

        rows.append({
            "threshold": round(th, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "accuracy": round(acc, 4),
            "predicted_churn_count": churn_count,
            "predicted_churn_pct": round(churn_pct, 2),
        })

    return pd.DataFrame(rows)


def assign_churn_risk_band(probability: float) -> str:
    """Assign an operational churn risk tier based on predicted probability.

    Operational Definitions:
    - Low Risk: probability < 0.20
    - Medium Risk: 0.20 <= probability < 0.50
    - High Risk: 0.50 <= probability < 0.75
    - Critical Risk: probability >= 0.75

    Note: These tiers represent operational workflow categories for business interventions,
    not calibrated probability guarantees.

    Args:
        probability: Float predicted probability score in [0.0, 1.0].

    Returns:
        Risk band label string.
    """
    if probability < 0.20:
        return "Low Risk"
    elif probability < 0.50:
        return "Medium Risk"
    elif probability < 0.75:
        return "High Risk"
    else:
        return "Critical Risk"


def generate_risk_band_summary(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """Summarize empirical performance across risk bands.

    Args:
        y_true: Ground truth target vector.
        y_prob: Predicted probabilities.

    Returns:
        pd.DataFrame summarizing total customers, churned customers, and actual churn rate by tier.
    """
    bands = [assign_churn_risk_band(p) for p in y_prob]
    df = pd.DataFrame({"risk_band": bands, "actual_churn": np.asarray(y_true)})

    summary = df.groupby("risk_band")["actual_churn"].agg(
        total_customers="count",
        actual_churn_count="sum",
        actual_churn_rate=lambda x: round(float(x.mean() * 100), 2),
    ).reset_index()

    order = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    summary["sort_key"] = summary["risk_band"].map(lambda x: order.index(x) if x in order else 99)
    return summary.sort_values(by="sort_key").drop(columns=["sort_key"]).reset_index(drop=True)


def generate_test_predictions_dataframe(
    customer_ids: Union[List[str], np.ndarray, pd.Series],
    y_true: Union[pd.Series, np.ndarray],
    lr_prob: np.ndarray,
    rf_prob: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Build unified test predictions DataFrame containing identifiers, probabilities, and risk tiers.

    Args:
        customer_ids: Array of customerID strings.
        y_true: Ground truth binary target vector.
        lr_prob: Logistic Regression predicted probabilities.
        rf_prob: Random Forest predicted probabilities.
        threshold: Standard decision threshold.

    Returns:
        pd.DataFrame formatted for test_predictions.csv export.
    """
    lr_risk_bands = [assign_churn_risk_band(p) for p in lr_prob]

    return pd.DataFrame({
        "customerID": list(customer_ids),
        "actual_churn": list(np.asarray(y_true)),
        "logistic_probability": np.round(lr_prob, 4),
        "logistic_prediction_0_50": (lr_prob >= threshold).astype(int),
        "logistic_risk_band": lr_risk_bands,
        "random_forest_probability": np.round(rf_prob, 4),
        "random_forest_prediction_0_50": (rf_prob >= threshold).astype(int),
    })
