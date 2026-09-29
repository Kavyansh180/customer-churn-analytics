"""Preprocessing and feature transformation pipeline for Customer Churn Analytics."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean_data import DEFAULT_PROCESSED_FILE
from src.features.feature_engineering import FeatureEngineer

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
DEFAULT_FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.csv"


def split_features_and_target(
    df: pd.DataFrame,
    id_col: str = "customerID",
    target_col: str = "Churn",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target variable from predictors and exclude high-cardinality ID column.

    Why customerID is excluded:
    - customerID is an arbitrary unique identifier with no generalized predictive value.
    - Including it would induce severe overfitting, data leakage, and artificial memorization.

    Args:
        df: Input DataFrame.
        id_col: Column name for customer identifier.
        target_col: Column name for target churn label.

    Returns:
        Tuple of (X DataFrame, y binary Series where No=0, Yes=1).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    drop_cols = [c for c in [id_col, target_col] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    # Convert binary target: 'No' -> 0, 'Yes' -> 1
    if df[target_col].dtype == "object":
        y = (df[target_col].str.strip().str.capitalize() == "Yes").astype(int)
    else:
        y = df[target_col].astype(int)

    return X, y


def perform_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing sets with stratified sampling.

    Why stratification is appropriate:
    - Churn is moderately imbalanced (~26.58% positive class).
    - Stratified splitting guarantees that the exact class proportion (73.42% retained, 26.58% churned)
      is preserved identically across both train and test splits, preventing sampling distortion.

    Args:
        X: Predictor DataFrame.
        y: Target Series.
        test_size: Proportion of dataset to include in test split (default: 0.20).
        random_state: Random seed for deterministic reproducibility.
        stratify: Whether to use stratified sampling based on y.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def identify_feature_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Automatically detect numerical and categorical feature columns.

    Args:
        df: DataFrame containing predictor features.

    Returns:
        Tuple of (numerical_columns_list, categorical_columns_list).
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessing_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    add_feature_engineering: bool = True,
) -> Pipeline:
    """Construct an end-to-end scikit-learn preprocessing pipeline.

    Architecture:
    1. FeatureEngineer (optional, enabled by default): creates `total_services` and `monthly_charges_diff`.
    2. ColumnTransformer:
       - Numerical Pipeline:
         * SimpleImputer(strategy='median')
         * StandardScaler()
       - Categorical Pipeline:
         * SimpleImputer(strategy='most_frequent')
         * OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Data Leakage Prevention:
    - The pipeline is constructed as a single unit and fitted strictly on X_train.
    - Statistics (mean/std for StandardScaler, medians for Imputers, categories for OneHotEncoder)
      are learned solely from training data and applied immutably to X_test.

    Args:
        num_cols: List of numerical column names.
        cat_cols: List of categorical column names.
        add_feature_engineering: Whether to include the FeatureEngineer transformer step.

    Returns:
        Scikit-Learn Pipeline instance.
    """
    # 1. Numerical sub-pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Categorical sub-pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3. Column Transformer
    transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # 4. Composite Pipeline
    if add_feature_engineering:
        pipeline = Pipeline(
            steps=[
                ("feat_eng", FeatureEngineer()),
                ("preprocessor", transformer),
            ]
        )
    else:
        pipeline = Pipeline(steps=[("preprocessor", transformer)])

    return pipeline


def extract_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """Extract human-readable transformed feature names from the fitted pipeline.

    Args:
        fitted_pipeline: A fitted scikit-learn Pipeline containing a ColumnTransformer.

    Returns:
        List of formatted feature name strings.
    """
    col_transformer = fitted_pipeline.named_steps.get("preprocessor")
    if col_transformer is None:
        raise ValueError("Pipeline does not contain a 'preprocessor' ColumnTransformer step.")
    
    raw_names = col_transformer.get_feature_names_out()
    return list(raw_names)


def save_feature_names(
    feature_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save extracted feature names to CSV for auditability and model interpretability.

    Args:
        feature_names: List of feature names.
        output_path: Target CSV path.

    Returns:
        Path to written file.
    """
    target = Path(output_path) if output_path else DEFAULT_FEATURE_NAMES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    
    df_names = pd.DataFrame({
        "feature_index": list(range(len(feature_names))),
        "feature_name": feature_names,
        "feature_group": ["numerical" if f.startswith("num__") else "categorical" for f in feature_names],
    })
    df_names.to_csv(target, index=False)
    return target


def save_preprocessor(
    preprocessor: Pipeline,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Serialize fitted preprocessing pipeline to disk using joblib.

    Args:
        preprocessor: Fitted scikit-learn Pipeline.
        output_path: Target path for the joblib artifact.

    Returns:
        Path to saved joblib file.
    """
    target = Path(output_path) if output_path else DEFAULT_PREPROCESSOR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, target)
    return target


def load_preprocessor(input_path: Optional[Union[str, Path]] = None) -> Pipeline:
    """Load serialized preprocessing pipeline from disk.

    Args:
        input_path: Optional path to joblib file.

    Returns:
        Fitted Pipeline instance.
    """
    target = Path(input_path) if input_path else DEFAULT_PREPROCESSOR_PATH
    if not target.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {target}")
    return joblib.load(target)


def save_split_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Persist train and test splits to data/processed for auditability.

    Args:
        X_train, X_test, y_train, y_test: Split DataFrames and Series.
        output_dir: Target directory.

    Returns:
        Dictionary of saved file paths.
    """
    target_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["Churn"] = y_train.values

    test_df = X_test.copy()
    test_df["Churn"] = y_test.values

    train_path = target_dir / "train.csv"
    test_path = target_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {"train": train_path, "test": test_path}


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """Execute end-to-end data splitting, feature engineering, and preprocessing pipeline."""
    # 1. Load cleaned data
    df = pd.read_csv(DEFAULT_PROCESSED_FILE)
    
    # 2. Split X and y (excluding customerID)
    X, y = split_features_and_target(df)
    
    # 3. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=True
    )
    
    # 4. Identify columns on engineered preview
    fe_preview = FeatureEngineer().transform(X_train)
    num_cols, cat_cols = identify_feature_types(fe_preview)
    
    # 5. Build Pipeline
    pipeline = build_preprocessing_pipeline(
        num_cols=num_cols,
        cat_cols=cat_cols,
        add_feature_engineering=True,
    )
    
    # 6. Fit strictly on training data (Leakage Prevention)
    pipeline.fit(X_train)
    
    # 7. Transform train and test
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    
    # 8. Extract & Save feature names
    feature_names = extract_feature_names(pipeline)
    names_path = save_feature_names(feature_names)
    
    # 9. Save preprocessor artifact
    model_path = save_preprocessor(pipeline)
    
    # 10. Persist train/test datasets
    split_paths = save_split_datasets(X_train, X_test, y_train, y_test)

    summary = {
        "raw_features_count": X.shape[1],
        "training_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "numerical_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "transformed_feature_count": len(feature_names),
        "X_train_transformed_shape": X_train_trans.shape,
        "X_test_transformed_shape": X_test_trans.shape,
        "preprocessor_path": str(model_path),
        "feature_names_path": str(names_path),
        "train_dataset_path": str(split_paths["train"]),
        "test_dataset_path": str(split_paths["test"]),
    }
    
    return summary


if __name__ == "__main__":
    res = run_preprocessing_pipeline()
    print("=" * 70)
    print(" CUSTOMER CHURN ANALYTICS - PREPROCESSING PIPELINE EXECUTION")
    print("=" * 70)
    print(f"  * Training Rows: {res['training_rows']:,} (80%)")
    print(f"  * Testing Rows: {res['test_rows']:,} (20%)")
    print(f"  * Train Churn Rate: {res['train_churn_rate']*100:.2f}%")
    print(f"  * Test Churn Rate: {res['test_churn_rate']*100:.2f}%")
    print(f"  * Numerical Features ({res['numerical_features_count']}): {res['numerical_columns']}")
    print(f"  * Categorical Features ({res['categorical_features_count']}): {res['categorical_columns']}")
    print(f"  * Transformed Feature Matrix: {res['transformed_feature_count']} features")
    print(f"  * Preprocessor saved to: {res['preprocessor_path']}")
    print(f"  * Feature names saved to: {res['feature_names_path']}")
    print("=" * 70)
