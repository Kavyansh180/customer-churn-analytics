"""Unit tests for Phase 6: Model Evaluation, Threshold Analysis, and Risk Banding."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import load_preprocessor
from src.models.model_utils import load_data_splits, load_model, DEFAULT_LR_PATH, DEFAULT_RF_PATH
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_ks_statistic,
    calculate_model_comparison_table,
)
from src.evaluation.threshold_analysis import (
    DEFAULT_THRESHOLDS,
    assign_churn_risk_band,
    evaluate_threshold_grid,
    generate_risk_band_summary,
    generate_test_predictions_dataframe,
)
from src.evaluation.evaluate_models import EVAL_ARTIFACTS_DIR


@pytest.fixture
def evaluation_data():
    """Load test data, preprocessor, and models to generate probabilities."""
    _, X_test, _, y_test = load_data_splits()
    preprocessor = load_preprocessor()
    lr = load_model(DEFAULT_LR_PATH)
    rf = load_model(DEFAULT_RF_PATH)

    X_test_trans = preprocessor.transform(X_test)
    lr_prob = lr.predict_proba(X_test_trans)[:, 1]
    rf_prob = rf.predict_proba(X_test_trans)[:, 1]

    return {
        "y_test": y_test.values,
        "lr_prob": lr_prob,
        "rf_prob": rf_prob,
        "n_samples": len(y_test),
    }


def test_evaluation_artifacts_exist():
    """1. Verify all generated Phase 6 CSV and image artifacts exist on disk."""
    expected_files = [
        "model_comparison.csv",
        "threshold_analysis.csv",
        "test_predictions.csv",
        "logistic_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "roc_curves.png",
        "precision_recall_curves.png",
        "logistic_calibration_curve.png",
    ]
    for fname in expected_files:
        p = EVAL_ARTIFACTS_DIR / fname
        assert p.exists(), f"Missing evaluation artifact: {fname}"
        assert p.stat().st_size > 0


def test_probabilities_between_zero_and_one(evaluation_data):
    """2. Verify predicted probability arrays are bounded in [0.0, 1.0]."""
    lr_prob = evaluation_data["lr_prob"]
    rf_prob = evaluation_data["rf_prob"]

    assert (lr_prob >= 0.0).all() and (lr_prob <= 1.0).all()
    assert (rf_prob >= 0.0).all() and (rf_prob <= 1.0).all()


def test_roc_auc_range(evaluation_data):
    """3. Verify ROC-AUC score is strictly bounded in (0.5, 1.0] for a trained classifier."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]
    rf_prob = evaluation_data["rf_prob"]

    lr_m = calculate_classification_metrics(y_test, lr_prob)
    rf_m = calculate_classification_metrics(y_test, rf_prob)

    assert 0.5 < lr_m["roc_auc"] <= 1.0
    assert 0.5 < rf_m["roc_auc"] <= 1.0
    assert lr_m["roc_auc"] > 0.80


def test_gini_equals_two_auc_minus_one(evaluation_data):
    """4. Verify Gini coefficient satisfies Gini = 2 * AUC - 1."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]

    m = calculate_classification_metrics(y_test, lr_prob)
    expected_gini = 2 * m["roc_auc"] - 1

    assert pytest.approx(m["gini"], abs=1e-3) == expected_gini


def test_ks_statistic_range(evaluation_data):
    """5. Verify KS statistic is between 0.0 and 1.0."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]

    ks_stat, ks_thresh = calculate_ks_statistic(y_test, lr_prob)
    assert 0.0 <= ks_stat <= 1.0
    assert 0.0 <= ks_thresh <= 1.0
    assert ks_stat > 0.40


def test_confusion_matrix_dimensions(evaluation_data):
    """6. Verify confusion matrix counts sum to exact test dataset length (1,407)."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]

    m = calculate_classification_metrics(y_test, lr_prob, threshold=0.50)
    total_cm = m["true_negatives"] + m["false_positives"] + m["false_negatives"] + m["true_positives"]

    assert total_cm == len(y_test)
    assert m["true_negatives"] == 918
    assert m["false_positives"] == 115
    assert m["false_negatives"] == 161
    assert m["true_positives"] == 213


def test_threshold_analysis_contains_requested_grid(evaluation_data):
    """7. Verify threshold grid evaluation includes all 17 specified thresholds."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]

    df_th = evaluate_threshold_grid(y_test, lr_prob, thresholds=DEFAULT_THRESHOLDS)
    assert len(df_th) == len(DEFAULT_THRESHOLDS)
    assert list(df_th["threshold"].values) == [round(t, 2) for t in DEFAULT_THRESHOLDS]


def test_precision_recall_f1_validity(evaluation_data):
    """8. Verify all computed precision, recall, and F1 scores fall in [0.0, 1.0]."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]

    df_th = evaluate_threshold_grid(y_test, lr_prob)
    for col in ["precision", "recall", "f1_score", "accuracy"]:
        assert (df_th[col] >= 0.0).all() and (df_th[col] <= 1.0).all()


def test_test_predictions_row_count(evaluation_data):
    """9. Verify test predictions table length matches test dataset length (1,407)."""
    p_csv = EVAL_ARTIFACTS_DIR / "test_predictions.csv"
    assert p_csv.exists()
    df_preds = pd.read_csv(p_csv)

    assert len(df_preds) == evaluation_data["n_samples"]
    assert "customerID" in df_preds.columns
    assert "logistic_risk_band" in df_preds.columns


def test_all_risk_bands_belong_to_valid_set(evaluation_data):
    """10. Verify all assigned churn risk bands match the 4 valid operational categories."""
    valid_bands = {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}
    lr_prob = evaluation_data["lr_prob"]
    assigned_bands = [assign_churn_risk_band(p) for p in lr_prob]

    assert set(assigned_bands).issubset(valid_bands)

    p_csv = EVAL_ARTIFACTS_DIR / "test_predictions.csv"
    df_preds = pd.read_csv(p_csv)
    assert set(df_preds["logistic_risk_band"].unique()).issubset(valid_bands)


def test_brier_score_range(evaluation_data):
    """11. Verify Brier score loss is bounded in [0.0, 1.0] and indicates good calibration."""
    y_test = evaluation_data["y_test"]
    lr_prob = evaluation_data["lr_prob"]

    m = calculate_classification_metrics(y_test, lr_prob)
    assert 0.0 <= m["brier_score"] <= 1.0
    assert m["brier_score"] < 0.20


def test_no_missing_predicted_probabilities(evaluation_data):
    """12. Verify zero missing or infinite values in prediction arrays."""
    lr_prob = evaluation_data["lr_prob"]
    rf_prob = evaluation_data["rf_prob"]

    assert not np.isnan(lr_prob).any()
    assert not np.isnan(rf_prob).any()
    assert not np.isinf(lr_prob).any()
    assert not np.isinf(rf_prob).any()
