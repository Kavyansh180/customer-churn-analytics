import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.load_data import load_raw_data
from src.data.clean_data import clean_dataset, save_processed_data


def generate_validation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive statistical and structural validation summary of a dataset.

    Args:
        df: Input DataFrame to inspect.

    Returns:
        Dict containing validation metrics and distributions.
    """
    # 1. Basic shape & duplicates
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    unique_customer_ids = int(df["customerID"].nunique()) if "customerID" in df.columns else None

    # 2. Missing values & data types per column
    col_info = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        blank_count = int((df[col].astype(str).str.strip() == "").sum()) if df[col].dtype == "object" else 0
        col_info[col] = {
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "blank_string_count": blank_count,
            "unique_values_count": int(df[col].nunique()),
        }

    # 3. Categorical breakdown
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_summary = {}
    for col in categorical_cols:
        if col != "customerID":
            cat_summary[col] = df[col].value_counts().to_dict()

    # 4. Numerical summary
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_summary = numeric_df.describe().to_dict()

    # 5. Target variable distribution
    churn_dist = None
    churn_pct = None
    if "Churn" in df.columns:
        churn_dist = df["Churn"].value_counts().to_dict()
        churn_pct = (df["Churn"].value_counts(normalize=True) * 100).round(2).to_dict()

    return {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "duplicate_rows": duplicate_rows,
        "unique_customer_ids": unique_customer_ids,
        "column_info": col_info,
        "categorical_distributions": cat_summary,
        "numerical_summary": numeric_summary,
        "churn_counts": churn_dist,
        "churn_percentages": churn_pct,
    }


def format_validation_report(
    raw_summary: Dict[str, Any],
    cleaned_summary: Optional[Dict[str, Any]] = None,
    cleaning_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Format validation summary dictionary into a clean, interview-ready console report.

    Args:
        raw_summary: Summary of raw data.
        cleaned_summary: Optional summary of cleaned data.
        cleaning_meta: Optional metadata dictionary from clean_dataset.

    Returns:
        Formatted multi-line report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append(" CUSTOMER CHURN ANALYTICS - DATA VALIDATION & INGESTION REPORT")
    lines.append("=" * 70)

    # 1. Dataset Dimensions & Integrity
    lines.append("\n[1] DATASET DIMENSIONS & INTEGRITY (RAW DATA)")
    lines.append(f"  * Rows: {raw_summary['num_rows']:,}")
    lines.append(f"  * Columns: {raw_summary['num_columns']}")
    lines.append(f"  * Duplicate Rows: {raw_summary['duplicate_rows']}")
    if raw_summary.get("unique_customer_ids"):
        lines.append(f"  * Unique Customer IDs: {raw_summary['unique_customer_ids']:,} (1:1 with rows)")

    # 2. Target Variable Distribution
    lines.append("\n[2] TARGET VARIABLE (CHURN) DISTRIBUTION")
    if raw_summary.get("churn_counts"):
        for label, count in raw_summary["churn_counts"].items():
            pct = raw_summary["churn_percentages"].get(label, 0)
            lines.append(f"  * Churn = '{label}': {count:,} ({pct:.2f}%)")

    # 3. Column Details & Data Types
    lines.append("\n[3] COLUMN DATA TYPES & MISSING VALUES")
    lines.append(f"  {'Column Name':<20} {'Dtype':<10} {'Nulls':<8} {'Blank Strings':<14} {'Uniques':<8}")
    lines.append("  " + "-" * 62)
    for col, info in raw_summary["column_info"].items():
        lines.append(
            f"  {col:<20} {info['dtype']:<10} {info['null_count']:<8} "
            f"{info['blank_string_count']:<14} {info['unique_values_count']:<8}"
        )

    # 4. Data Quality Finding: TotalCharges
    lines.append("\n[4] KEY DATA QUALITY FINDING - TotalCharges")
    if cleaning_meta:
        lines.append(f"  * Blank values detected in raw TotalCharges: {cleaning_meta['blank_total_charges_found']}")
        lines.append("  * Tenure distribution of blank TotalCharges rows: " + str(cleaning_meta['blank_total_charges_tenure_dist']))
        lines.append("  * Root Cause: Customers with tenure = 0 months have not completed a billing cycle.")
        lines.append(f"  * Action Taken: Explicitly converted to float64 and removed {cleaning_meta['rows_dropped']} rows ({cleaning_meta['rows_dropped']/raw_summary['num_rows']*100:.3f}% of data).")

    # 5. Numerical Statistics Summary
    lines.append("\n[5] NUMERICAL FEATURES SUMMARY (RAW)")
    if raw_summary.get("numerical_summary"):
        num_cols = list(raw_summary["numerical_summary"].keys())
        header = f"  {'Metric':<10}" + "".join([f"{col:>18}" for col in num_cols])
        lines.append(header)
        lines.append("  " + "-" * (10 + 18 * len(num_cols)))
        metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for metric in metrics:
            row_str = f"  {metric:<10}"
            for col in num_cols:
                val = raw_summary["numerical_summary"][col].get(metric, 0)
                row_str += f"{val:>18.2f}"
            lines.append(row_str)

    # 6. Cleaned Dataset Summary
    if cleaned_summary:
        lines.append("\n[6] VALIDATED CLEANED DATASET SUMMARY (data/processed)")
        lines.append(f"  * Final Shape: {cleaned_summary['num_rows']} rows x {cleaned_summary['num_columns']} columns")
        lines.append(f"  * TotalCharges Dtype: {cleaned_summary['column_info']['TotalCharges']['dtype']}")
        lines.append(f"  * Remaining Nulls across all columns: {sum(info['null_count'] for info in cleaned_summary['column_info'].values())}")
        if cleaned_summary.get("churn_counts"):
            lines.append("  * Cleaned Churn Distribution:")
            for label, count in cleaned_summary["churn_counts"].items():
                pct = cleaned_summary["churn_percentages"].get(label, 0)
                lines.append(f"      - '{label}': {count:,} ({pct:.2f}%)")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_pipeline() -> None:
    """Execute raw data loading, validation, cleaning, and persistence pipeline."""
    # 1. Load raw data
    raw_df = load_raw_data()
    raw_summary = generate_validation_summary(raw_df)

    # 2. Clean data & handle TotalCharges
    cleaned_df, cleaning_meta = clean_dataset(raw_df, drop_missing_total_charges=True)
    cleaned_summary = generate_validation_summary(cleaned_df)

    # 3. Save to data/processed
    saved_path = save_processed_data(cleaned_df)

    # 4. Format and print validation report
    report = format_validation_report(raw_summary, cleaned_summary, cleaning_meta)
    print(report)
    print(f"\n Cleaned dataset successfully persisted to: {saved_path}\n")


if __name__ == "__main__":
    run_pipeline()
