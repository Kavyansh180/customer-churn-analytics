"""Model evaluation, metrics, threshold analysis, and churn risk banding modules."""

from src.evaluation.metrics import (
    calculate_ks_statistic,
    calculate_classification_metrics,
    calculate_model_comparison_table,
)
from src.evaluation.threshold_analysis import (
    DEFAULT_THRESHOLDS,
    evaluate_threshold_grid,
    assign_churn_risk_band,
    generate_risk_band_summary,
    generate_test_predictions_dataframe,
)
from src.evaluation.evaluate_models import (
    run_evaluation_pipeline,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_calibration_curve,
    EVAL_ARTIFACTS_DIR,
)

__all__ = [
    "calculate_ks_statistic",
    "calculate_classification_metrics",
    "calculate_model_comparison_table",
    "DEFAULT_THRESHOLDS",
    "evaluate_threshold_grid",
    "assign_churn_risk_band",
    "generate_risk_band_summary",
    "generate_test_predictions_dataframe",
    "run_evaluation_pipeline",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_calibration_curve",
    "EVAL_ARTIFACTS_DIR",
]
