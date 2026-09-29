"""Model utilities for persistence, feature coefficient extraction, and metadata tracking."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
DEFAULT_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
DEFAULT_PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
DEFAULT_FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.csv"
DEFAULT_LR_PATH = ARTIFACTS_DIR / "logistic_regression.joblib"
DEFAULT_RF_PATH = ARTIFACTS_DIR / "random_forest.joblib"
DEFAULT_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
DEFAULT_COEF_PATH = ARTIFACTS_DIR / "logistic_coefficients.csv"


def load_data_splits(
    train_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    target_col: str = "Churn",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train and test CSV datasets and separate predictors from target.

    Args:
        train_path: Optional custom path to train.csv.
        test_path: Optional custom path to test.csv.
        target_col: Target column name.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    tr_path = Path(train_path) if train_path else DEFAULT_TRAIN_PATH
    te_path = Path(test_path) if test_path else DEFAULT_TEST_PATH

    if not tr_path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {tr_path}")
    if not te_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {te_path}")

    train_df = pd.read_csv(tr_path)
    test_df = pd.read_csv(te_path)

    # Exclude customerID if present
    drop_cols = [c for c in ["customerID", target_col] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols).copy()
    y_train = train_df[target_col].astype(int).copy()

    drop_cols_test = [c for c in ["customerID", target_col] if c in test_df.columns]
    X_test = test_df.drop(columns=drop_cols_test).copy()
    y_test = test_df[target_col].astype(int).copy()

    return X_train, X_test, y_train, y_test


def save_model(model: Any, output_path: Union[str, Path]) -> Path:
    """Serialize a trained model estimator using joblib.

    Args:
        model: Trained scikit-learn model object.
        output_path: Target path for the joblib file.

    Returns:
        Path to saved joblib file.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(model_path: Union[str, Path]) -> Any:
    """Load a serialized model from disk using joblib.

    Args:
        model_path: Path to joblib file.

    Returns:
        Fitted model object.
    """
    target = Path(model_path)
    if not target.exists():
        raise FileNotFoundError(f"Model artifact not found at: {target}")
    return joblib.load(target)


def extract_logistic_coefficients(
    model: LogisticRegression,
    feature_names_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Extract feature coefficients and calculate odds ratios for Logistic Regression.

    Formula:
        Odds Ratio = exp(beta)

    Interpretation:
        - beta > 0 (Odds Ratio > 1): Positive association; increases the log-odds (and odds) of churn.
        - beta < 0 (Odds Ratio < 1): Negative association; decreases the log-odds (and odds) of churn.
        - beta = 0 (Odds Ratio = 1): No linear association with churn log-odds.

    Args:
        model: Fitted LogisticRegression model.
        feature_names_path: Optional path to feature_names.csv.

    Returns:
        pd.DataFrame sorted by absolute coefficient magnitude.
    """
    fn_path = Path(feature_names_path) if feature_names_path else DEFAULT_FEATURE_NAMES_PATH
    if fn_path.exists():
        df_fn = pd.read_csv(fn_path)
        feature_names = df_fn["feature_name"].tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]

    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)

    df_coef = pd.DataFrame({
        "feature_index": list(range(len(feature_names))),
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": odds_ratios,
        "abs_coefficient": np.abs(coefficients),
        "direction": [
            "Increases Churn Risk (Positive)" if c > 0 else "Decreases Churn Risk (Negative)"
            for c in coefficients
        ],
        "feature_group": [
            "numerical" if f.startswith("num__") else "categorical" for f in feature_names
        ],
    })

    return df_coef.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)


def save_logistic_coefficients(
    coef_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save Logistic Regression coefficients and odds ratios to CSV.

    Args:
        coef_df: DataFrame of extracted coefficients.
        output_path: Target CSV path.

    Returns:
        Path to saved CSV.
    """
    target = Path(output_path) if output_path else DEFAULT_COEF_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(target, index=False)
    return target


def save_model_metadata(
    metadata: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save model training metadata and configurations to JSON.

    Args:
        metadata: Metadata dictionary.
        output_path: Target JSON path.

    Returns:
        Path to saved JSON.
    """
    target = Path(output_path) if output_path else DEFAULT_METADATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return target
