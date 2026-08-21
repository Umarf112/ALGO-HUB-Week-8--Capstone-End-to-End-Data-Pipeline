"""
Utility Functions for Logging, Directory Setup, and Data I/O
"""
import logging
import os
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_DIR, DATA_DIR

def setup_logger(name: str = "DataPipeline") -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def ensure_directories():
    """Ensure all required project directories exist."""
    dirs = [DATA_DIR / "raw", DATA_DIR / "processed", MODEL_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at {path}")
    df = pd.read_csv(path)
    return df

def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH):
    """Save processed feature matrix to CSV."""
    ensure_directories()
    df.to_csv(path, index=False)
    print(f"Processed dataset successfully saved to {path} ({len(df)} rows, {df.shape[1]} columns)")
