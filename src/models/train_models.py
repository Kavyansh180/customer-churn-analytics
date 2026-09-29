"""Model training module for Logistic Regression (primary interpretable) and Random Forest (benchmark)."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import load_preprocessor
from src.models.model_utils import (
    load_data_splits,
    save_model,
    load_model,
    extract_logistic_coefficients,
    save_logistic_coefficients,
    save_model_metadata,
    DEFAULT_LR_PATH,
    DEFAULT_RF_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_COEF_PATH,
)


def train_logistic_regression(
    X_train_trans: np.ndarray,
    y_train: Union[pd.Series, np.ndarray],
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """Train primary interpretable Logistic Regression classification model.

    Configuration:
        - max_iter=1000: Guarantees convergence on standardized feature spaces.
        - random_state=42: Ensures deterministic, reproducible optimization.

    Args:
        X_train_trans: Transformed training feature matrix (N x D).
        y_train: Binary training target vector.
        max_iter: Maximum solver iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted LogisticRegression estimator.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train_trans, y_train)
    return model


def train_random_forest(
    X_train_trans: np.ndarray,
    y_train: Union[pd.Series, np.ndarray],
    n_estimators: int = 300,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Train benchmark non-linear Random Forest classification model.

    Configuration:
        - n_estimators=300: Provides robust ensemble variance reduction.
        - random_state=42: Guarantees tree splitting reproducibility.
        - n_jobs=-1: Parallelizes tree construction across all available CPU cores.

    Args:
        X_train_trans: Transformed training feature matrix (N x D).
        y_train: Binary training target vector.
        n_estimators: Number of decision trees.
        random_state: Random seed for reproducibility.
        n_jobs: CPU core parallelization count (-1 uses all cores).

    Returns:
        Fitted RandomForestClassifier estimator.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train_trans, y_train)
    return model


def run_training_pipeline() -> Dict[str, Any]:
    """Execute end-to-end model training, sanity verification, and artifact serialization."""
    # 1. Load data splits
    X_train, X_test, y_train, y_test = load_data_splits()

    # 2. Load preprocessor and transform features
    preprocessor = load_preprocessor()
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Dimensionality verification
    n_train_rows, n_features = X_train_trans.shape
    n_test_rows, _ = X_test_trans.shape

    # 3. Train Logistic Regression
    lr_model = train_logistic_regression(X_train_trans, y_train)

    # 4. Train Random Forest
    rf_model = train_random_forest(X_train_trans, y_train)

    # 5. Sanity Checks & Predictions
    lr_preds = lr_model.predict(X_test_trans)
    lr_probs = lr_model.predict_proba(X_test_trans)[:, 1]

    rf_preds = rf_model.predict(X_test_trans)
    rf_probs = rf_model.predict_proba(X_test_trans)[:, 1]

    # Verification assertions
    assert len(lr_preds) == len(y_test), "Logistic Regression prediction length mismatch."
    assert len(rf_preds) == len(y_test), "Random Forest prediction length mismatch."
    assert (lr_probs >= 0.0).all() and (lr_probs <= 1.0).all(), "LR probabilities out of [0, 1] range."
    assert (rf_probs >= 0.0).all() and (rf_probs <= 1.0).all(), "RF probabilities out of [0, 1] range."
    assert not np.isnan(lr_probs).any(), "NaN found in LR probabilities."
    assert not np.isnan(rf_probs).any(), "NaN found in RF probabilities."

    # 6. Save Model Artifacts
    lr_path = save_model(lr_model, DEFAULT_LR_PATH)
    rf_path = save_model(rf_model, DEFAULT_RF_PATH)

    # 7. Extract & Save Logistic Regression Coefficients
    coef_df = extract_logistic_coefficients(lr_model)
    coef_path = save_logistic_coefficients(coef_df, DEFAULT_COEF_PATH)

    # 8. Compile and Save Metadata
    metadata = {
        "project": "Customer Churn Prediction & Retention Analytics System",
        "phase": 5,
        "phase_name": "Model Training",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_definition": {
            "name": "Churn",
            "negative_class": 0,
            "positive_class": 1,
            "mapping": {"No": 0, "Yes": 1},
        },
        "dataset_dimensions": {
            "training_rows": n_train_rows,
            "test_rows": n_test_rows,
            "transformed_feature_count": n_features,
            "training_churn_positive_count": int(y_train.sum()),
            "training_retained_negative_count": int((y_train == 0).sum()),
            "training_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "models": {
            "logistic_regression": {
                "model_class": "LogisticRegression",
                "role": "Primary Interpretable Model",
                "artifact_path": str(lr_path.relative_to(PROJECT_ROOT)) if lr_path.is_relative_to(PROJECT_ROOT) else str(lr_path),
                "parameters": {
                    "max_iter": lr_model.max_iter,
                    "random_state": lr_model.random_state,
                    "solver": lr_model.solver,
                    "penalty": lr_model.penalty,
                    "C": lr_model.C,
                },
                "training_status": "Success",
                "coefficients_artifact_path": str(coef_path.relative_to(PROJECT_ROOT)) if coef_path.is_relative_to(PROJECT_ROOT) else str(coef_path),
            },
            "random_forest": {
                "model_class": "RandomForestClassifier",
                "role": "Nonlinear Benchmark Model",
                "artifact_path": str(rf_path.relative_to(PROJECT_ROOT)) if rf_path.is_relative_to(PROJECT_ROOT) else str(rf_path),

                "parameters": {
                    "n_estimators": rf_model.n_estimators,
                    "random_state": rf_model.random_state,
                    "n_jobs": rf_model.n_jobs,
                    "criterion": rf_model.criterion,
                    "max_depth": rf_model.max_depth,
                },
                "training_status": "Success",
            },
        },
    }
    meta_path = save_model_metadata(metadata, DEFAULT_METADATA_PATH)

    return {
        "training_rows": n_train_rows,
        "test_rows": n_test_rows,
        "transformed_feature_count": n_features,
        "lr_training_status": "Trained & Verified",
        "rf_training_status": "Trained & Verified",
        "lr_parameters": metadata["models"]["logistic_regression"]["parameters"],
        "rf_parameters": metadata["models"]["random_forest"]["parameters"],
        "lr_artifact_path": str(lr_path),
        "rf_artifact_path": str(rf_path),
        "coefficients_path": str(coef_path),
        "metadata_path": str(meta_path),
        "top_churn_drivers": coef_df.head(5)[["feature", "coefficient", "odds_ratio", "direction"]].to_dict(orient="records"),
        "top_retention_drivers": coef_df.tail(5)[["feature", "coefficient", "odds_ratio", "direction"]].to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = run_training_pipeline()
    print("=" * 70)
    print(" CUSTOMER CHURN ANALYTICS - PHASE 5: MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"  * Training Rows: {result['training_rows']:,}")
    print(f"  * Test Rows: {result['test_rows']:,}")
    print(f"  * Transformed Feature Count: {result['transformed_feature_count']}")
    print(f"  * Logistic Regression Status: {result['lr_training_status']}")
    print(f"  * Random Forest Status: {result['rf_training_status']}")
    print(f"  * Logistic Regression Parameters: {result['lr_parameters']}")
    print(f"  * Random Forest Parameters: {result['rf_parameters']}")
    print(f"  * Logistic Regression Artifact: {result['lr_artifact_path']}")
    print(f"  * Random Forest Artifact: {result['rf_artifact_path']}")
    print(f"  * Logistic Coefficients CSV: {result['coefficients_path']}")
    print(f"  * Model Metadata JSON: {result['metadata_path']}")
    print("=" * 70)
