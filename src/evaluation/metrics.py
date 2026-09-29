"""Evaluation metrics computation module for binary customer churn classification."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_ks_statistic(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
) -> Tuple[float, float]:
    """Calculate Kolmogorov-Smirnov (KS) statistic and separation threshold.

    KS measures the maximum vertical distance between the cumulative distribution
    functions of churned (positive) and retained (negative) probability scores.

    Args:
        y_true: Binary ground truth array {0, 1}.
        y_prob: Predicted probability array in [0.0, 1.0].

    Returns:
        Tuple of (ks_statistic, threshold_at_max_separation).
    """
    df_ks = pd.DataFrame({"y": np.asarray(y_true), "prob": np.asarray(y_prob)})
    df_ks = df_ks.sort_values(by="prob", ascending=False).reset_index(drop=True)

    total_pos = (df_ks["y"] == 1).sum()
    total_neg = (df_ks["y"] == 0).sum()

    if total_pos == 0 or total_neg == 0:
        return 0.0, 0.5

    df_ks["cum_pos"] = (df_ks["y"] == 1).cumsum() / total_pos
    df_ks["cum_neg"] = (df_ks["y"] == 0).cumsum() / total_neg
    df_ks["ks_diff"] = np.abs(df_ks["cum_pos"] - df_ks["cum_neg"])

    max_idx = df_ks["ks_diff"].idxmax()
    ks_stat = float(df_ks.loc[max_idx, "ks_diff"])
    ks_threshold = float(df_ks.loc[max_idx, "prob"])

    return round(ks_stat, 4), round(ks_threshold, 4)


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: Binary ground truth array.
        y_prob: Predicted probability array.
        threshold: Decision boundary for binary classification.

    Returns:
        Dictionary containing metric values.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    gini = float(2 * roc_auc - 1)
    ks_stat, ks_thresh = calculate_ks_statistic(y_true_arr, y_prob_arr)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

    return {
        "threshold": threshold,
        "roc_auc": round(roc_auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_threshold": round(ks_thresh, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calculate_model_comparison_table(
    y_true: Union[pd.Series, np.ndarray],
    model_probabilities: Dict[str, np.ndarray],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Generate comparative summary DataFrame across multiple classification models.

    Args:
        y_true: Ground truth vector.
        model_probabilities: Dict mapping model name string to probability array.
        threshold: Evaluation decision threshold.

    Returns:
        pd.DataFrame formatted for model comparison export.
    """
    rows = []
    for model_name, y_prob in model_probabilities.items():
        m = calculate_classification_metrics(y_true, y_prob, threshold=threshold)
        rows.append({
            "model": model_name,
            "roc_auc": m["roc_auc"],
            "gini": m["gini"],
            "ks": m["ks_statistic"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1_score"],
            "brier_score": m["brier_score"],
        })
    return pd.DataFrame(rows)
