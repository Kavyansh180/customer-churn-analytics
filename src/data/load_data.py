"""Data loading module for Customer Churn Prediction & Retention Analytics System."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RAW_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def locate_raw_data_file(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Locate the raw CSV dataset file in the specified or default raw data directory.

    Args:
        data_dir: Optional directory to search in. Defaults to data/raw/.

    Returns:
        Path to the located CSV file.

    Raises:
        FileNotFoundError: If no CSV file is found in the target directory.
    """
    target_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {target_dir}")

    # Check for default filename first
    if DEFAULT_RAW_FILE.exists() and (data_dir is None or Path(data_dir) == RAW_DATA_DIR):
        return DEFAULT_RAW_FILE

    # Otherwise look for any .csv file in the target directory
    csv_files = list(target_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in raw data directory: {target_dir}")

    return csv_files[0]


def load_raw_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the raw customer churn dataset into a Pandas DataFrame.

    Args:
        file_path: Optional exact path to the CSV file. If None, auto-locates in data/raw/.

    Returns:
        pd.DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the specified or auto-located file does not exist.
        ValueError: If the loaded file is empty.
    """
    target_path = Path(file_path) if file_path else locate_raw_data_file()

    if not target_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")

    df = pd.read_csv(target_path)

    if df.empty:
        raise ValueError(f"Loaded dataset from {target_path} is empty.")

    return df
