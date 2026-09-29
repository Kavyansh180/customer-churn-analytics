"""Data cleaning and preprocessing module for Phase 2."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PROCESSED_FILE = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"

# Expected core columns in the IBM Telco Churn dataset
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def clean_dataset(
    df: pd.DataFrame,
    drop_missing_total_charges: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Clean and validate raw Telco Customer Churn DataFrame.

    Key cleaning steps:
    1. Validates presence of expected schema columns.
    2. Identifies whitespace/blank strings in 'TotalCharges'.
    3. Converts 'TotalCharges' from object/string to float64.
    4. Handles invalid 'TotalCharges' values:
       - Blank values correspond to brand-new accounts with tenure = 0 months.
       - If `drop_missing_total_charges=True`, these 11 rows (~0.156% of records) are dropped
         because they lack billing history and have zero tenure.
       - If `False`, they are imputed with 0.0.

    Args:
        df: Input raw DataFrame.
        drop_missing_total_charges: Whether to drop rows with unconvertible TotalCharges (tenure=0).

    Returns:
        Tuple of (cleaned_df, cleaning_metadata_dict).
    """
    cleaned_df = df.copy()

    # 1. Schema check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in cleaned_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    # 2. Check blank/whitespace values in TotalCharges
    raw_blank_mask = cleaned_df["TotalCharges"].astype(str).str.strip() == ""
    blank_count = int(raw_blank_mask.sum())

    # Collect details of blank rows before conversion
    blank_tenure_dist = cleaned_df.loc[raw_blank_mask, "tenure"].value_counts().to_dict()

    # 3. Convert TotalCharges to numeric (coercing whitespace to NaN)
    cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")

    rows_before = len(cleaned_df)
    rows_dropped = 0

    if drop_missing_total_charges:
        # Drop rows where TotalCharges is NaN (tenure == 0 accounts)
        cleaned_df = cleaned_df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
        rows_dropped = rows_before - len(cleaned_df)
    else:
        # Impute with 0.0 for tenure == 0 accounts
        cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(0.0)

    metadata = {
        "rows_raw": rows_before,
        "rows_cleaned": len(cleaned_df),
        "columns_count": len(cleaned_df.columns),
        "blank_total_charges_found": blank_count,
        "blank_total_charges_tenure_dist": blank_tenure_dist,
        "rows_dropped": rows_dropped,
        "drop_missing_total_charges_used": drop_missing_total_charges,
        "total_charges_dtype": str(cleaned_df["TotalCharges"].dtype),
    }

    return cleaned_df, metadata


def save_processed_data(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save cleaned DataFrame to data/processed directory without modifying raw data.

    Args:
        df: Cleaned DataFrame.
        output_path: Optional custom path for output CSV. Defaults to data/processed/telco_churn_cleaned.csv.

    Returns:
        Path to saved processed file.
    """
    target_path = Path(output_path) if output_path else DEFAULT_PROCESSED_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path
