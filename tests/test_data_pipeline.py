"""Unit tests for Phase 2: Data Ingestion, Cleaning, and Validation."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.load_data import load_raw_data, locate_raw_data_file, DEFAULT_RAW_FILE
from src.data.clean_data import clean_dataset, save_processed_data, EXPECTED_COLUMNS
from src.data.validate_data import generate_validation_summary


def test_locate_raw_data_file_exists():
    """Verify raw dataset file location exists."""
    file_path = locate_raw_data_file()
    assert file_path.exists(), f"Raw data file not found at: {file_path}"
    assert file_path.suffix == ".csv"


def test_load_raw_data_success():
    """Verify loading raw dataset returns valid non-empty DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert df.shape[1] == 21


def test_load_raw_data_missing_file_raises_error(tmp_path):
    """Verify appropriate error is raised when dataset file is missing."""
    non_existent = tmp_path / "missing_data.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(file_path=non_existent)


def test_expected_columns_present():
    """Verify all expected domain schema columns exist in the loaded data."""
    df = load_raw_data()
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_churn_target_exists_and_valid():
    """Verify target column 'Churn' exists with expected binary categories."""
    df = load_raw_data()
    assert "Churn" in df.columns
    unique_churn = sorted(df["Churn"].dropna().unique().tolist())
    assert unique_churn == ["No", "Yes"]


def test_total_charges_blank_handling_and_conversion():
    """Verify TotalCharges contains 11 whitespace values and is properly converted."""
    raw_df = load_raw_data()
    
    # Check blank count in raw
    blank_mask = raw_df["TotalCharges"].astype(str).str.strip() == ""
    assert blank_mask.sum() == 11

    # Clean with default drop mode
    cleaned_df, meta = clean_dataset(raw_df, drop_missing_total_charges=True)
    
    assert meta["blank_total_charges_found"] == 11
    assert meta["rows_dropped"] == 11
    assert len(cleaned_df) == 7032
    assert pd.api.types.is_float_dtype(cleaned_df["TotalCharges"])
    assert cleaned_df["TotalCharges"].isnull().sum() == 0


def test_clean_dataset_impute_mode():
    """Verify TotalCharges imputation mode retains all 7043 rows and replaces NaN with 0.0."""
    raw_df = load_raw_data()
    cleaned_df, meta = clean_dataset(raw_df, drop_missing_total_charges=False)
    
    assert len(cleaned_df) == 7043
    assert meta["rows_dropped"] == 0
    assert cleaned_df["TotalCharges"].isnull().sum() == 0
    assert pd.api.types.is_float_dtype(cleaned_df["TotalCharges"])


def test_generate_validation_summary():
    """Verify validation summary produces expected statistical attributes."""
    raw_df = load_raw_data()
    summary = generate_validation_summary(raw_df)
    
    assert summary["num_rows"] == 7043
    assert summary["num_columns"] == 21
    assert summary["duplicate_rows"] == 0
    assert summary["unique_customer_ids"] == 7043
    assert "No" in summary["churn_counts"]
    assert "Yes" in summary["churn_counts"]
    assert summary["churn_counts"]["No"] == 5174
    assert summary["churn_counts"]["Yes"] == 1869


def test_processed_data_persistence(tmp_path):
    """Verify cleaned dataset persists to CSV properly."""
    raw_df = load_raw_data()
    cleaned_df, _ = clean_dataset(raw_df, drop_missing_total_charges=True)
    
    out_file = tmp_path / "test_processed.csv"
    saved = save_processed_data(cleaned_df, output_path=out_file)
    
    assert saved.exists()
    reloaded = pd.read_csv(saved)
    assert len(reloaded) == 7032
    assert reloaded["TotalCharges"].dtype == float
