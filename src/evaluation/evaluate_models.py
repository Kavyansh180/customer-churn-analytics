"""Evaluation pipeline and visualization generator for Logistic Regression and Random Forest."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.data.clean_data import DEFAULT_PROCESSED_FILE
from src.features.preprocessing import load_preprocessor
from src.models.model_utils import (
    load_data_splits,
    load_model,
    DEFAULT_LR_PATH,
    DEFAULT_RF_PATH,
)
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_model_comparison_table,
    calculate_ks_statistic,
)
from src.evaluation.threshold_analysis import (
    evaluate_threshold_grid,
    generate_risk_band_summary,
    generate_test_predictions_dataframe,
)

EVAL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "evaluation"


def get_test_customer_ids() -> np.ndarray:
    """Retrieve original customerID values for test partition aligned with random_state=42."""
    df_clean = pd.read_csv(DEFAULT_PROCESSED_FILE)
    y = (df_clean["Churn"] == "Yes").astype(int)
    X = df_clean.drop(columns=["customerID", "Churn"])

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return df_clean.loc[X_test.index, "customerID"].values


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Generate and save publication-grade annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    annot_matrix = np.array([
        [f"True Neg (TN)\n{tn:,}", f"False Pos (FP)\n{fp:,}"],
        [f"False Neg (FN)\n{fn:,}", f"True Pos (TP)\n{tp:,}"],
    ])

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Retained (0)", "Predicted Churn (1)"],
        yticklabels=["Actual Retained (0)", "Actual Churn (1)"],
        annot_kws={"size": 11, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix @ 0.50 Threshold\n{model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    lr_prob: np.ndarray,
    rf_prob: np.ndarray,
    output_path: Path,
) -> None:
    """Generate combined ROC curve plot for Logistic Regression and Random Forest."""
    lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_true, rf_prob)

    lr_auc = roc_auc_score(y_true, lr_prob)
    rf_auc = roc_auc_score(y_true, rf_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, color="#2b5c8f", lw=2.2, label=f"Logistic Regression (AUC = {lr_auc:.4f})")
    plt.plot(rf_fpr, rf_tpr, color="#d9534f", lw=2.2, linestyle="--", label=f"Random Forest (AUC = {rf_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.2, linestyle=":", label="Random Classifier (AUC = 0.5000)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    lr_prob: np.ndarray,
    rf_prob: np.ndarray,
    output_path: Path,
) -> None:
    """Generate combined Precision-Recall curve plot."""
    lr_prec, lr_rec, _ = precision_recall_curve(y_true, lr_prob)
    rf_prec, rf_rec, _ = precision_recall_curve(y_true, rf_prob)
    baseline_rate = float(y_true.mean())

    plt.figure(figsize=(8, 6))
    plt.plot(lr_rec, lr_prec, color="#2b5c8f", lw=2.2, label="Logistic Regression")
    plt.plot(rf_rec, rf_prec, color="#d9534f", lw=2.2, linestyle="--", label="Random Forest")
    plt.axhline(baseline_rate, color="#888888", lw=1.2, linestyle=":", label=f"Baseline Churn Rate ({baseline_rate*100:.1f}%)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (True Positive Rate)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Precision-Recall Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    lr_prob: np.ndarray,
    output_path: Path,
) -> None:
    """Generate probability reliability calibration diagram for Logistic Regression."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, lr_prob, n_bins=10, strategy="uniform"
    )
    brier = brier_score_loss(y_true, lr_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color="#2b5c8f", lw=2, label=f"Logistic Regression (Brier = {brier:.4f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Empirical Fraction of Positives (Actual Churn)", fontsize=11)
    plt.title("Probability Calibration Curve (Reliability Diagram)\nLogistic Regression", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_evaluation_pipeline() -> Dict[str, Any]:
    """Execute complete Phase 6 evaluation pipeline, export CSV tables, and render plots."""
    EVAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and models
    _, X_test, _, y_test = load_data_splits()
    preprocessor = load_preprocessor()
    lr_model = load_model(DEFAULT_LR_PATH)
    rf_model = load_model(DEFAULT_RF_PATH)

    # 2. Transform test data
    X_test_trans = preprocessor.transform(X_test)
    y_test_arr = y_test.values

    # 3. Generate probabilities
    lr_prob = lr_model.predict_proba(X_test_trans)[:, 1]
    rf_prob = rf_model.predict_proba(X_test_trans)[:, 1]

    # 4. Model Comparison Table
    df_comparison = calculate_model_comparison_table(
        y_test_arr,
        {"Logistic Regression": lr_prob, "Random Forest": rf_prob},
        threshold=0.50,
    )
    comparison_path = EVAL_ARTIFACTS_DIR / "model_comparison.csv"
    df_comparison.to_csv(comparison_path, index=False)

    # 5. Threshold Analysis for Logistic Regression
    df_threshold = evaluate_threshold_grid(y_test_arr, lr_prob)
    threshold_path = EVAL_ARTIFACTS_DIR / "threshold_analysis.csv"
    df_threshold.to_csv(threshold_path, index=False)

    # 6. Test Predictions with Risk Bands
    customer_ids = get_test_customer_ids()
    df_predictions = generate_test_predictions_dataframe(
        customer_ids, y_test_arr, lr_prob, rf_prob, threshold=0.50
    )
    predictions_path = EVAL_ARTIFACTS_DIR / "test_predictions.csv"
    df_predictions.to_csv(predictions_path, index=False)

    # 7. Risk Band Summary
    df_risk_summary = generate_risk_band_summary(y_test_arr, lr_prob)

    # 8. Render and Save Evaluation Plots
    p_cm_lr = EVAL_ARTIFACTS_DIR / "logistic_confusion_matrix.png"
    p_cm_rf = EVAL_ARTIFACTS_DIR / "random_forest_confusion_matrix.png"
    p_roc = EVAL_ARTIFACTS_DIR / "roc_curves.png"
    p_pr = EVAL_ARTIFACTS_DIR / "precision_recall_curves.png"
    p_cal = EVAL_ARTIFACTS_DIR / "logistic_calibration_curve.png"

    plot_confusion_matrix(y_test_arr, (lr_prob >= 0.50).astype(int), "Logistic Regression", p_cm_lr)
    plot_confusion_matrix(y_test_arr, (rf_prob >= 0.50).astype(int), "Random Forest", p_cm_rf)
    plot_roc_curves(y_test_arr, lr_prob, rf_prob, p_roc)
    plot_precision_recall_curves(y_test_arr, lr_prob, rf_prob, p_pr)
    plot_calibration_curve(y_test_arr, lr_prob, p_cal)

    lr_metrics = calculate_classification_metrics(y_test_arr, lr_prob, threshold=0.50)
    rf_metrics = calculate_classification_metrics(y_test_arr, rf_prob, threshold=0.50)

    return {
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "model_comparison_table": df_comparison.to_dict(orient="records"),
        "threshold_count": len(df_threshold),
        "test_prediction_count": len(df_predictions),
        "risk_band_summary": df_risk_summary.to_dict(orient="records"),
        "artifacts_created": [
            str(comparison_path),
            str(threshold_path),
            str(predictions_path),
            str(p_cm_lr),
            str(p_cm_rf),
            str(p_roc),
            str(p_pr),
            str(p_cal),
        ],
    }


if __name__ == "__main__":
    res = run_evaluation_pipeline()
    print("=" * 70)
    print(" CUSTOMER CHURN ANALYTICS - PHASE 6: MODEL EVALUATION COMPLETE")
    print("=" * 70)
    print("\nMODEL PERFORMANCE COMPARISON (@ 0.50 Threshold):")
    print(pd.DataFrame(res["model_comparison_table"]).to_string(index=False))
    print("\nOPERATIONAL RISK BAND DISTRIBUTION (Logistic Regression):")
    print(pd.DataFrame(res["risk_band_summary"]).to_string(index=False))
    print("\nArtifacts Saved to artifacts/evaluation/:")
    for a in res["artifacts_created"]:
        print(f"  * {Path(a).name}")
    print("=" * 70)
