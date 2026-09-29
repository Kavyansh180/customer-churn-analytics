"""Exploratory Data Analysis (EDA) module for Customer Churn Analytics."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.clean_data import DEFAULT_PROCESSED_FILE

EDA_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "eda"
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
KEY_CATEGORICAL_COLS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "PaperlessBilling",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "MultipleLines",
    "StreamingTV",
    "StreamingMovies",
    "gender",
]


def load_cleaned_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load cleaned dataset from data/processed directory.

    Args:
        file_path: Optional custom path to cleaned CSV.

    Returns:
        pd.DataFrame containing the cleaned data.
    """
    target = Path(file_path) if file_path else DEFAULT_PROCESSED_FILE
    if not target.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {target}")
    return pd.read_csv(target)


def calculate_churn_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall churn counts and percentages."""
    if "Churn" not in df.columns:
        raise ValueError("DataFrame does not contain 'Churn' column.")
    
    total = len(df)
    counts = df["Churn"].value_counts().to_dict()
    churn_yes = counts.get("Yes", 0)
    churn_no = counts.get("No", 0)
    
    return {
        "total_customers": total,
        "churn_yes_count": churn_yes,
        "churn_no_count": churn_no,
        "churn_rate_pct": round((churn_yes / total) * 100, 2) if total > 0 else 0.0,
        "retention_rate_pct": round((churn_no / total) * 100, 2) if total > 0 else 0.0,
    }


def calculate_numerical_summary(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate standard summary statistics for numerical features."""
    target_cols = cols or NUMERICAL_COLS
    summary = df[target_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary["median"] = df[target_cols].median()
    summary["IQR"] = summary["75%"] - summary["25%"]
    return summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "IQR"]]


def calculate_numerical_churn_breakdown(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate group-level statistics for numerical features split by Churn."""
    target_cols = cols or NUMERICAL_COLS
    records = []
    for col in target_cols:
        for churn_val in ["No", "Yes"]:
            sub = df[df["Churn"] == churn_val][col]
            q25 = sub.quantile(0.25)
            q75 = sub.quantile(0.75)
            records.append({
                "feature": col,
                "churn": churn_val,
                "count": len(sub),
                "mean": round(sub.mean(), 2),
                "median": round(sub.median(), 2),
                "std": round(sub.std(), 2),
                "min": round(sub.min(), 2),
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "iqr": round(q75 - q25, 2),
                "max": round(sub.max(), 2),
            })
    return pd.DataFrame(records)


def calculate_categorical_churn_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calculate churn counts, total customers, and churn rate (%) for categories of a feature."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    agg = df.groupby(col)["Churn"].agg(
        total_customers="count",
        churn_count=lambda x: int((x == "Yes").sum()),
        churn_rate=lambda x: round(float((x == "Yes").mean() * 100), 2)
    ).reset_index()
    
    agg["retention_count"] = agg["total_customers"] - agg["churn_count"]
    agg["pct_of_all_customers"] = round((agg["total_customers"] / len(df)) * 100, 2)
    return agg.sort_values(by="churn_rate", ascending=False).reset_index(drop=True)


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for numerical features plus binary churn indicator."""
    corr_df = df[NUMERICAL_COLS + ["SeniorCitizen"]].copy()
    corr_df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    return corr_df.corr()


def check_numerical_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Inspect numerical features using 1.5 * IQR rule for potential outliers."""
    target_cols = cols or NUMERICAL_COLS
    results = []
    for col in target_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        results.append({
            "feature": col,
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outliers),
            "outlier_pct": round((len(outliers) / len(df)) * 100, 2),
        })
    return pd.DataFrame(results)


def generate_all_eda_plots(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Generate and save publication-grade, interview-ready EDA visualizations.

    Args:
        df: Cleaned DataFrame.
        output_dir: Output directory for saving images.

    Returns:
        List of Paths to created image files.
    """
    out_dir = Path(output_dir) if output_dir else EDA_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    # Visual theme setup
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
    })

    # Color palette
    colors_binary = {"No": "#2b5c8f", "Yes": "#d9534f"}

    # -------------------------------------------------------------
    # 1. Target Distribution Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df["Churn"].value_counts()
    
    # Bar plot
    bars = ax[0].bar(
        churn_counts.index,
        churn_counts.values,
        color=[colors_binary[k] for k in churn_counts.index],
        width=0.5,
    )
    ax[0].set_title("Target Class Distribution (Counts)", fontsize=13, fontweight="bold", pad=12)
    ax[0].set_xlabel("Churn", fontsize=11)
    ax[0].set_ylabel("Number of Customers", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width() / 2, h + 80, f"{h:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax[0].set_ylim(0, max(churn_counts.values) * 1.15)

    # Donut plot
    wedges, texts, autotexts = ax[1].pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=[colors_binary[k] for k in churn_counts.index],
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=11),
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax[1].set_title("Target Class Proportion (%)", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    p1 = out_dir / "01_target_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_plots.append(p1)

    # -------------------------------------------------------------
    # 2. Univariate Numerical Distributions
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for i, col in enumerate(NUMERICAL_COLS):
        # Histogram & KDE
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color="#2b5c8f", bins=30)
        axes[i, 0].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        axes[i, 0].set_xlabel(col, fontsize=10)
        axes[i, 0].set_ylabel("Count", fontsize=10)

        # Boxplot
        sns.boxplot(x=df[col], ax=axes[i, 1], color="#4a90e2", fliersize=3)
        axes[i, 1].set_title(f"Boxplot of {col}", fontsize=12, fontweight="bold")
        axes[i, 1].set_xlabel(col, fontsize=10)

    plt.tight_layout()
    p2 = out_dir / "02_univariate_numerical_distributions.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_plots.append(p2)

    # -------------------------------------------------------------
    # 3. Churn vs Numerical Features (Boxplots & KDE)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, col in enumerate(NUMERICAL_COLS):
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            ax=axes[i],
            hue="Churn",
            palette=colors_binary,
            legend=False,
            order=["No", "Yes"],
            width=0.4,
        )
        axes[i].set_title(f"{col} by Churn Status", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Customer Churned", fontsize=11)
        axes[i].set_ylabel(col, fontsize=11)

    plt.tight_layout()
    p3 = out_dir / "03_churn_vs_numerical_boxplots.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    saved_plots.append(p3)

    # -------------------------------------------------------------
    # 4. Churn Rates Across Key Account/Contract Features
    # -------------------------------------------------------------
    key_features = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for i, col in enumerate(key_features):
        rate_df = calculate_categorical_churn_rates(df, col)
        sns.barplot(
            data=rate_df,
            x=col,
            y="churn_rate",
            ax=axes[i],
            hue=col,
            palette="Blues_r",
            legend=False,
        )
        axes[i].set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Churn Rate (%)", fontsize=10)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylim(0, max(rate_df["churn_rate"]) * 1.25)
        axes[i].axhline(26.58, color="#d9534f", linestyle="--", linewidth=1.2, label="Overall Avg (26.58%)")
        axes[i].legend(loc="upper right", fontsize=9)
        
        # Add labels
        for p in axes[i].patches:
            h = p.get_height()
            axes[i].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1.0), ha="center", fontsize=9, fontweight="bold")
        
        if col in ["PaymentMethod", "Contract"]:
            axes[i].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    p4 = out_dir / "04_churn_by_key_account_features.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    saved_plots.append(p4)

    # -------------------------------------------------------------
    # 5. Services & Support Churn Rates
    # -------------------------------------------------------------
    services = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
    service_rates = []
    for s in services:
        rates = calculate_categorical_churn_rates(df, s)
        for _, row in rates.iterrows():
            service_rates.append({
                "Service": s,
                "Option": str(row[s]),
                "Churn Rate (%)": row["churn_rate"],
            })
    service_df = pd.DataFrame(service_rates)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=service_df,
        x="Service",
        y="Churn Rate (%)",
        hue="Option",
        palette={"No": "#d9534f", "Yes": "#2b5c8f", "No internet service": "#95a5a6"},
    )
    plt.title("Churn Rate Across Value-Added Tech & Streaming Services", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=11)
    plt.xlabel("Service Category", fontsize=11)
    plt.axhline(26.58, color="black", linestyle="--", linewidth=1, label="Overall Baseline (26.58%)")
    plt.legend(title="Service Status", loc="upper right")
    plt.tight_layout()
    p5 = out_dir / "05_services_churn_rates.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    saved_plots.append(p5)

    # -------------------------------------------------------------
    # 6. Correlation Heatmap
    # -------------------------------------------------------------
    corr = calculate_correlation_matrix(df)
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True,
    )
    plt.title("Correlation Matrix of Numerical Features & Churn", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    p6 = out_dir / "06_correlation_heatmap.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    saved_plots.append(p6)

    return saved_plots


def run_eda_pipeline() -> None:
    """Run end-to-end EDA pipeline and export plots and summary reports."""
    df = load_cleaned_data()
    print("=" * 70)
    print(" CUSTOMER CHURN ANALYTICS - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    summary = calculate_churn_summary(df)
    print(f"\nDataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Overall Churn: {summary['churn_yes_count']:,} churned ({summary['churn_rate_pct']}%), {summary['churn_no_count']:,} retained ({summary['retention_rate_pct']}%)")

    print("\nGenerating publication-grade plots in artifacts/eda/...")
    plots = generate_all_eda_plots(df)
    for p in plots:
        print(f"  * Created plot: {p.name}")

    print("\nOutlier Assessment (IQR Rule):")
    outliers = check_numerical_outliers(df)
    print(outliers.to_string(index=False))

    print("\nEDA Pipeline execution complete.\n")


if __name__ == "__main__":
    run_eda_pipeline()
