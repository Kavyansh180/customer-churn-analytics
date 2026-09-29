"""Unit tests for Phase 3: Exploratory Data Analysis (EDA) functions."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.eda import (
    load_cleaned_data,
    calculate_churn_summary,
    calculate_numerical_summary,
    calculate_numerical_churn_breakdown,
    calculate_categorical_churn_rates,
    calculate_correlation_matrix,
    check_numerical_outliers,
    generate_all_eda_plots,
)


def test_load_cleaned_data_success():
    """Verify cleaned dataset loads with expected 7032 rows and 21 columns."""
    df = load_cleaned_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7032
    assert df.shape[1] == 21
    assert "Churn" in df.columns


def test_calculate_churn_summary():
    """Verify overall churn rate and class counts."""
    df = load_cleaned_data()
    summary = calculate_churn_summary(df)
    
    assert summary["total_customers"] == 7032
    assert summary["churn_yes_count"] == 1869
    assert summary["churn_no_count"] == 5163
    assert pytest.approx(summary["churn_rate_pct"], 0.01) == 26.58
    assert pytest.approx(summary["retention_rate_pct"], 0.01) == 73.42


def test_calculate_numerical_summary():
    """Verify summary statistics for numerical features."""
    df = load_cleaned_data()
    num_summary = calculate_numerical_summary(df)
    
    assert "tenure" in num_summary.index
    assert "MonthlyCharges" in num_summary.index
    assert "TotalCharges" in num_summary.index
    
    # Check bounds
    assert num_summary.loc["tenure", "min"] == 1.0
    assert num_summary.loc["tenure", "max"] == 72.0
    assert pytest.approx(num_summary.loc["MonthlyCharges", "min"], 0.01) == 18.25
    assert pytest.approx(num_summary.loc["TotalCharges", "max"], 0.01) == 8684.80


def test_calculate_numerical_churn_breakdown():
    """Verify grouped numeric statistics by Churn."""
    df = load_cleaned_data()
    breakdown = calculate_numerical_churn_breakdown(df)
    
    assert len(breakdown) == 6  # 3 features * 2 churn categories
    
    # Verify tenure of churned customers is lower than retained
    tenure_no = breakdown[(breakdown["feature"] == "tenure") & (breakdown["churn"] == "No")]["mean"].values[0]
    tenure_yes = breakdown[(breakdown["feature"] == "tenure") & (breakdown["churn"] == "Yes")]["mean"].values[0]
    assert tenure_yes < tenure_no


def test_calculate_categorical_churn_rates_contract():
    """Verify categorical churn rate calculation for Contract type."""
    df = load_cleaned_data()
    contract_rates = calculate_categorical_churn_rates(df, "Contract")
    
    assert len(contract_rates) == 3
    assert "Contract" in contract_rates.columns
    assert "churn_rate" in contract_rates.columns
    
    # Month-to-month should have the highest churn rate
    top_churn_contract = contract_rates.iloc[0]["Contract"]
    assert top_churn_contract == "Month-to-month"
    assert pytest.approx(contract_rates.iloc[0]["churn_rate"], 0.01) == 42.71


def test_calculate_correlation_matrix():
    """Verify correlation matrix output contains numerical columns and binary churn."""
    df = load_cleaned_data()
    corr = calculate_correlation_matrix(df)
    
    expected_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_Binary"]
    for col in expected_cols:
        assert col in corr.columns
        assert corr.loc[col, col] == 1.0


def test_check_numerical_outliers():
    """Verify 1.5 * IQR outlier inspection reports zero outliers across bounded columns."""
    df = load_cleaned_data()
    outlier_df = check_numerical_outliers(df)
    
    assert len(outlier_df) == 3
    assert outlier_df["outlier_count"].sum() == 0


def test_generate_all_eda_plots(tmp_path):
    """Verify all 6 publication plots are generated and saved correctly."""
    df = load_cleaned_data()
    plot_paths = generate_all_eda_plots(df, output_dir=tmp_path)
    
    assert len(plot_paths) == 6
    for p in plot_paths:
        assert p.exists()
        assert p.stat().st_size > 0
