"""
Automated Exploratory Data Analysis (EDA) Engine.
Computes real dataset statistics, missing value reports, correlations, and distribution metrics.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH, EDA_SUMMARY_PATH, NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL


def generate_eda_report(df: pd.DataFrame, save_path: Path = EDA_SUMMARY_PATH) -> dict:
    """
    Analyzes raw dataset and computes comprehensive statistical metrics.
    Saves JSON summary report to disk.
    """
    df_clean = df.copy()
    
    # Identify space strings in string columns
    space_strings_map = {}
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            space_count = (df_clean[col].astype(str).str.strip() == "").sum()
            if space_count > 0:
                space_strings_map[col] = int(space_count)

    # Convert TotalCharges space strings to NaN for accurate numeric analysis
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges_numeric"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    else:
        df_clean["TotalCharges_numeric"] = np.nan

    num_cols_analysis = ["tenure", "MonthlyCharges", "TotalCharges_numeric"]

    # Basic Info
    summary = {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "duplicate_rows": int(df.duplicated().sum()),
        "space_strings": space_strings_map,
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
        "data_types": {col: str(df[col].dtype) for col in df.columns},
    }

    # Target Distribution
    if TARGET_COL in df.columns:
        target_counts = df[TARGET_COL].value_counts().to_dict()
        target_prop = df[TARGET_COL].value_counts(normalize=True).to_dict()
        summary["target_distribution"] = {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "proportions": {str(k): float(np.round(v, 4)) for k, v in target_prop.items()},
        }

    # Numerical Column Statistics
    num_stats = {}
    for col in num_cols_analysis:
        if col in df_clean.columns:
            s = df_clean[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers_count = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()
            
            num_stats[col] = {
                "count": int(len(s)),
                "mean": float(np.round(s.mean(), 2)),
                "std": float(np.round(s.std(), 2)),
                "min": float(np.round(s.min(), 2)),
                "25%": float(np.round(q1, 2)),
                "50%": float(np.round(s.median(), 2)),
                "75%": float(np.round(q3, 2)),
                "max": float(np.round(s.max(), 2)),
                "skewness": float(np.round(s.skew(), 2)),
                "outliers_iqr_count": int(outliers_count),
            }
    summary["numerical_statistics"] = num_stats

    # Categorical Column Summaries
    cat_stats = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            top_counts = df[col].value_counts().head(5).to_dict()
            cat_stats[col] = {
                "unique_count": int(df[col].nunique()),
                "top_categories": {str(k): int(v) for k, v in top_counts.items()},
            }
    summary["categorical_summaries"] = cat_stats

    # Numerical Correlations (including numeric target)
    if TARGET_COL in df_clean.columns:
        df_clean["Target_Encoded"] = (df_clean[TARGET_COL] == "Yes").astype(int)
        corr_cols = num_cols_analysis + ["Target_Encoded"]
        corr_df = df_clean[corr_cols].corr().round(4)
        summary["correlation_matrix"] = corr_df.to_dict()

    # Save to disk
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Automated EDA summary saved to {save_path}")

    return summary


if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_DATA_PATH)
    res = generate_eda_report(df_raw)
    print("EDA Report computed successfully.")
