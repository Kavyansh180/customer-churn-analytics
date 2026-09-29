"""Unit tests for Phase 5: Model Training, Persistence, and Sanity Checks."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import load_preprocessor
from src.models.model_utils import (
    load_data_splits,
    load_model,
    extract_logistic_coefficients,
    DEFAULT_LR_PATH,
    DEFAULT_RF_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_COEF_PATH,
    DEFAULT_FEATURE_NAMES_PATH,
)
from src.models.train_models import train_logistic_regression, train_random_forest


@pytest.fixture
def dataset_splits():
    """Load train/test split data fixture."""
    return load_data_splits()


@pytest.fixture
def preprocessor():
    """Load fitted preprocessor fixture."""
    return load_preprocessor()


def test_target_is_binary(dataset_splits):
    """1. Verify target variable in train and test is binary {0, 1}."""
    _, _, y_train, y_test = dataset_splits
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_train_test_dimensions(dataset_splits):
    """2. Verify train and test dimensions match expected Phase 4 split."""
    X_train, X_test, y_train, y_test = dataset_splits
    assert len(X_train) == 5625
    assert len(X_test) == 1407
    assert len(y_train) == 5625
    assert len(y_test) == 1407
    assert X_train.shape[1] == 19
    assert X_test.shape[1] == 19


def test_preprocessing_artifact_loads(preprocessor):
    """3. Verify preprocessor artifact loads and transforms features to 47 dimensions."""
    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_logistic_regression_model_loads():
    """4. Verify Logistic Regression artifact exists and loads as a fitted model."""
    assert DEFAULT_LR_PATH.exists()
    model = load_model(DEFAULT_LR_PATH)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert model.coef_.shape == (1, 47)


def test_random_forest_model_loads():
    """5. Verify Random Forest artifact exists and loads as a fitted ensemble."""
    assert DEFAULT_RF_PATH.exists()
    model = load_model(DEFAULT_RF_PATH)
    assert isinstance(model, RandomForestClassifier)
    assert len(model.estimators_) == 300
    assert model.n_features_in_ == 47


def test_prediction_lengths_are_correct(dataset_splits, preprocessor):
    """6. Verify prediction array lengths equal y_test length (1,407)."""
    _, X_test, _, y_test = dataset_splits
    X_test_trans = preprocessor.transform(X_test)

    lr = load_model(DEFAULT_LR_PATH)
    rf = load_model(DEFAULT_RF_PATH)

    lr_preds = lr.predict(X_test_trans)
    rf_preds = rf.predict(X_test_trans)

    assert len(lr_preds) == len(y_test)
    assert len(rf_preds) == len(y_test)


def test_probabilities_between_zero_and_one(dataset_splits, preprocessor):
    """7. Verify predicted churn probabilities fall strictly within [0.0, 1.0]."""
    _, X_test, _, _ = dataset_splits
    X_test_trans = preprocessor.transform(X_test)

    lr = load_model(DEFAULT_LR_PATH)
    rf = load_model(DEFAULT_RF_PATH)

    lr_probs = lr.predict_proba(X_test_trans)[:, 1]
    rf_probs = rf.predict_proba(X_test_trans)[:, 1]

    assert (lr_probs >= 0.0).all() and (lr_probs <= 1.0).all()
    assert (rf_probs >= 0.0).all() and (rf_probs <= 1.0).all()


def test_no_nan_probabilities(dataset_splits, preprocessor):
    """8. Verify no NaN or infinite values in predicted probabilities or classes."""
    _, X_test, _, _ = dataset_splits
    X_test_trans = preprocessor.transform(X_test)

    lr = load_model(DEFAULT_LR_PATH)
    rf = load_model(DEFAULT_RF_PATH)

    lr_probs = lr.predict_proba(X_test_trans)[:, 1]
    rf_probs = rf.predict_proba(X_test_trans)[:, 1]

    assert not np.isnan(lr_probs).any()
    assert not np.isnan(rf_probs).any()
    assert not np.isinf(lr_probs).any()
    assert not np.isinf(rf_probs).any()


def test_transformed_feature_count_matches_feature_names(dataset_splits, preprocessor):
    """9. Verify transformed feature count matches feature_names.csv (47 features)."""
    X_train, _, _, _ = dataset_splits
    X_train_trans = preprocessor.transform(X_train)

    assert DEFAULT_FEATURE_NAMES_PATH.exists()
    df_names = pd.read_csv(DEFAULT_FEATURE_NAMES_PATH)

    assert X_train_trans.shape[1] == 47
    assert len(df_names) == 47


def test_model_artifacts_and_metadata_exist():
    """10. Verify all model artifacts, metadata JSON, and coefficient CSV files exist."""
    assert DEFAULT_LR_PATH.exists()
    assert DEFAULT_RF_PATH.exists()
    assert DEFAULT_METADATA_PATH.exists()
    assert DEFAULT_COEF_PATH.exists()

    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["dataset_dimensions"]["training_rows"] == 5625
    assert meta["dataset_dimensions"]["test_rows"] == 1407
    assert meta["dataset_dimensions"]["transformed_feature_count"] == 47
    assert "logistic_regression" in meta["models"]
    assert "random_forest" in meta["models"]
