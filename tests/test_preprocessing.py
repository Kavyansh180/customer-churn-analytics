"""Unit tests for Phase 4: Preprocessing and Feature Engineering."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.clean_data import DEFAULT_PROCESSED_FILE
from src.features.feature_engineering import FeatureEngineer, engineer_features
from src.features.preprocessing import (
    split_features_and_target,
    perform_train_test_split,
    identify_feature_types,
    build_preprocessing_pipeline,
    extract_feature_names,
    save_preprocessor,
    load_preprocessor,
    DEFAULT_PREPROCESSOR_PATH,
    DEFAULT_FEATURE_NAMES_PATH,
)


@pytest.fixture
def sample_data():
    """Load cleaned dataset fixture."""
    return pd.read_csv(DEFAULT_PROCESSED_FILE)


def test_target_conversion(sample_data):
    """Verify target column conversion into binary 0/1 integer format."""
    X, y = split_features_and_target(sample_data)
    assert set(y.unique()) == {0, 1}
    assert y.dtype in [np.int64, np.int32, int]
    assert len(y) == len(sample_data)
    # Check that 'Yes' mapped to 1 and 'No' mapped to 0
    assert (y == 1).sum() == (sample_data["Churn"] == "Yes").sum()
    assert (y == 0).sum() == (sample_data["Churn"] == "No").sum()


def test_customer_id_exclusion(sample_data):
    """Verify customerID is excluded from feature set X to prevent data leakage/overfitting."""
    X, y = split_features_and_target(sample_data)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert X.shape[1] == 19  # 21 total - 1 ID - 1 target


def test_train_test_split_sizes(sample_data):
    """Verify 80/20 train/test split partitions data correctly."""
    X, y = split_features_and_target(sample_data)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.20, random_state=42, stratify=True)
    
    assert len(X_train) == 5625
    assert len(X_test) == 1407
    assert len(X_train) + len(X_test) == 7032
    assert len(y_train) == 5625
    assert len(y_test) == 1407


def test_stratification_ratios(sample_data):
    """Verify stratification preserves exact class proportions across train and test sets."""
    X, y = split_features_and_target(sample_data)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.20, random_state=42, stratify=True)
    
    overall_churn_rate = y.mean()
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    assert pytest.approx(train_churn_rate, abs=1e-3) == overall_churn_rate
    assert pytest.approx(test_churn_rate, abs=1e-3) == overall_churn_rate
    assert pytest.approx(train_churn_rate, abs=1e-3) == 0.2658


def test_feature_engineering_transformer(sample_data):
    """Verify FeatureEngineer transformer creates total_services and monthly_charges_diff."""
    X, _ = split_features_and_target(sample_data)
    fe = FeatureEngineer()
    X_fe = fe.transform(X)

    assert "total_services" in X_fe.columns
    assert "monthly_charges_diff" in X_fe.columns
    assert X_fe["total_services"].between(0, 8).all()
    assert not X_fe["monthly_charges_diff"].isnull().any()


def test_preprocessing_pipeline_fit_and_transform(sample_data):
    """Verify pipeline fits on X_train and transforms train and test to 47 features."""
    X, y = split_features_and_target(sample_data)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y, test_size=0.20, random_state=42, stratify=True)

    fe_preview = FeatureEngineer().transform(X_train)
    num_cols, cat_cols = identify_feature_types(fe_preview)

    pipeline = build_preprocessing_pipeline(num_cols=num_cols, cat_cols=cat_cols, add_feature_engineering=True)
    pipeline.fit(X_train)

    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    assert X_train_trans.shape == (5625, 47)
    assert X_test_trans.shape == (1407, 47)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_numerical_scaling(sample_data):
    """Verify numerical features are standardized to ~0 mean and ~1 std on training set."""
    X, y = split_features_and_target(sample_data)
    X_train, _, _, _ = perform_train_test_split(X, y, test_size=0.20, random_state=42, stratify=True)

    fe_preview = FeatureEngineer().transform(X_train)
    num_cols, cat_cols = identify_feature_types(fe_preview)

    pipeline = build_preprocessing_pipeline(num_cols=num_cols, cat_cols=cat_cols, add_feature_engineering=True)
    pipeline.fit(X_train)

    X_train_trans = pipeline.transform(X_train)
    # First 6 columns are numerical
    num_matrix = X_train_trans[:, :6]
    means = np.mean(num_matrix, axis=0)
    stds = np.std(num_matrix, axis=0)

    for m in means:
        assert pytest.approx(m, abs=1e-2) == 0.0
    for s in stds:
        assert pytest.approx(s, abs=1e-2) == 1.0


def test_unknown_categorical_values_handled(sample_data):
    """Verify handle_unknown='ignore' safely encodes unseen categories as all zeros without raising errors."""
    X, y = split_features_and_target(sample_data)
    X_train, X_test, _, _ = perform_train_test_split(X, y, test_size=0.20, random_state=42, stratify=True)

    fe_preview = FeatureEngineer().transform(X_train)
    num_cols, cat_cols = identify_feature_types(fe_preview)

    pipeline = build_preprocessing_pipeline(num_cols=num_cols, cat_cols=cat_cols, add_feature_engineering=True)
    pipeline.fit(X_train)

    # Introduce novel category in test data
    X_novel = X_test.copy().head(5)
    X_novel.loc[X_novel.index[0], "PaymentMethod"] = "Crypto Payment (Unseen)"
    X_novel.loc[X_novel.index[1], "InternetService"] = "Satellite (Unseen)"

    # Should transform gracefully with 0s for unknown categories
    transformed = pipeline.transform(X_novel)
    assert transformed.shape == (5, 47)
    assert not np.isnan(transformed).any()


def test_preprocessor_serialization_and_reloading(tmp_path, sample_data):
    """Verify serialized preprocessor joblib can be reloaded and yields identical output."""
    X, y = split_features_and_target(sample_data)
    X_train, X_test, _, _ = perform_train_test_split(X, y, test_size=0.20, random_state=42, stratify=True)

    fe_preview = FeatureEngineer().transform(X_train)
    num_cols, cat_cols = identify_feature_types(fe_preview)

    pipeline = build_preprocessing_pipeline(num_cols=num_cols, cat_cols=cat_cols, add_feature_engineering=True)
    pipeline.fit(X_train)

    out_file = tmp_path / "preprocessor_test.joblib"
    save_preprocessor(pipeline, output_path=out_file)

    reloaded = load_preprocessor(input_path=out_file)
    original_out = pipeline.transform(X_test)
    reloaded_out = reloaded.transform(X_test)

    np.testing.assert_array_almost_equal(original_out, reloaded_out)


def test_feature_names_csv_matches_preprocessor(sample_data):
    """Verify feature_names.csv matches extracted names from preprocessor artifact."""
    assert DEFAULT_PREPROCESSOR_PATH.exists()
    assert DEFAULT_FEATURE_NAMES_PATH.exists()

    pipeline = load_preprocessor(DEFAULT_PREPROCESSOR_PATH)
    extracted_names = extract_feature_names(pipeline)
    df_names = pd.read_csv(DEFAULT_FEATURE_NAMES_PATH)

    assert len(extracted_names) == 47
    assert len(df_names) == 47
    assert df_names["feature_name"].tolist() == extracted_names
