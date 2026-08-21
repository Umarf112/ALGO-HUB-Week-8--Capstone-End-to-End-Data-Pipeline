"""
Central Configuration for AlgoHub Week 8 Capstone Data Pipeline
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "Telco-Customer-Churn-Processed.csv"
MODEL_DIR = BASE_DIR / "models"
PIPELINE_SAVE_PATH = MODEL_DIR / "pipeline.joblib"
PREPROCESSOR_SAVE_PATH = MODEL_DIR / "preprocessor.joblib"
EDA_SUMMARY_PATH = MODEL_DIR / "eda_summary.json"
MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.json"

# Column Schema Definitions
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]

# Random Seed for Reproducibility
RANDOM_STATE = 42
