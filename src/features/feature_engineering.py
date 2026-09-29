"""Feature engineering module for Customer Churn Analytics."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer for engineering customer churn features.

    Engineered Features:
    1. `total_services` (int): Count of active subscribed services with value == 'Yes'.
       Rationale: Captures overall engagement depth and ecosystem lock-in.
       EDA demonstrated customers with 7-8 services churn at ~5-12%, vs ~44% for 0 services.
    
    2. `monthly_charges_diff` (float): Difference between current MonthlyCharges and
       historical average monthly spend (TotalCharges / max(tenure, 1)).
       Rationale: Positive values indicate recent price increases or expiring promotional
       discounts, creating price friction. Negative values indicate plan downgrades.
    """

    def __init__(
        self,
        add_service_count: bool = True,
        add_monthly_diff: bool = True,
        service_cols: Optional[List[str]] = None,
    ):
        self.add_service_count = add_service_count
        self.add_monthly_diff = add_monthly_diff
        self.service_cols = service_cols or SERVICE_COLUMNS

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit transformer (stateless, returns self)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame by appending engineered features.

        Args:
            X: Input DataFrame containing predictor features.

        Returns:
            pd.DataFrame with new engineered columns.
        """
        X_out = X.copy()

        # 1. Total Subscribed Services Count
        if self.add_service_count:
            # Check existing service columns
            available_services = [col for col in self.service_cols if col in X_out.columns]
            if available_services:
                X_out["total_services"] = X_out[available_services].apply(
                    lambda row: sum(1 for v in row if str(v).strip().lower() == "yes"),
                    axis=1,
                )

        # 2. Monthly Charges Difference vs Historical Average
        if self.add_monthly_diff:
            if "TotalCharges" in X_out.columns and "tenure" in X_out.columns and "MonthlyCharges" in X_out.columns:
                # Use np.maximum to strictly prevent division by zero for edge cases
                safe_tenure = np.maximum(X_out["tenure"].astype(float), 1.0)
                avg_monthly = X_out["TotalCharges"].astype(float) / safe_tenure
                X_out["monthly_charges_diff"] = X_out["MonthlyCharges"].astype(float) - avg_monthly

        return X_out


def engineer_features(
    df: pd.DataFrame,
    add_service_count: bool = True,
    add_monthly_diff: bool = True,
) -> pd.DataFrame:
    """Convenience functional interface for applying feature engineering."""
    fe = FeatureEngineer(
        add_service_count=add_service_count,
        add_monthly_diff=add_monthly_diff,
    )
    return fe.transform(df)
