"""
Pipeline Builder and Model Training Engine.
Constructs full scikit-learn Pipeline with custom transformers, trains multiple genuine ML models,
evaluates metrics (Accuracy, Precision, Recall, F1, ROC-AUC), selects best model, and serializes with joblib.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import (
    RAW_DATA_PATH,
    PIPELINE_SAVE_PATH,
    PREPROCESSOR_SAVE_PATH,
    MODEL_METRICS_PATH,
    PROCESSED_DATA_PATH,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    ID_COL,
    RANDOM_STATE,
)
from src.custom_transformers import (
    IDColumnDropper,
    DataTypeCoercer,
    MissingValueHandler,
    OutlierCapper,
    FeatureEngTransformer,
    CategoricalEncoder,
    FeatureScaler,
)
from src.utils import ensure_directories, load_raw_data, save_processed_data


def build_preprocessor() -> Pipeline:
    """
    Constructs the custom scikit-learn preprocessing pipeline.
    Steps:
    1. IDColumnDropper: Drops customerID if present.
    2. DataTypeCoercer: Converts string space numbers (TotalCharges) to NaN/float64.
    3. MissingValueHandler: Imputes missing numeric (medians) and categorical (modes).
    4. OutlierCapper: IQR-based capping on numeric columns.
    5. FeatureEngTransformer: Generates domain-specific features (TenureYears, HasAddonsCount, etc.).
    6. CategoricalEncoder: One-hot encodes all categorical columns.
    7. FeatureScaler: StandardScales numeric columns.
    """
    # Expanded numeric columns after feature engineering
    all_numeric_cols = NUMERIC_COLS + ["TenureYears", "AvgMonthlyCostPerTenure", "HasAddonsCount"]

    preprocessor = Pipeline([
        ("id_dropper", IDColumnDropper(id_cols=[ID_COL])),
        ("type_coercer", DataTypeCoercer(target_cols=["TotalCharges"])),
        ("missing_imputer", MissingValueHandler(numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS)),
        ("outlier_capper", OutlierCapper(numeric_cols=NUMERIC_COLS, factor=1.5)),
        ("feature_engineering", FeatureEngTransformer()),
        ("cat_encoder", CategoricalEncoder(categorical_cols=CATEGORICAL_COLS)),
        ("scaler", FeatureScaler(numeric_cols=all_numeric_cols)),
    ])
    return preprocessor


def train_and_evaluate_pipeline(raw_df: pd.DataFrame) -> dict:
    """
    Splits data, fits preprocessor, transforms features, trains candidate ML models,
    evaluates genuine metrics, selects best model, and serializes full pipeline.
    """
    ensure_directories()
    
    # Clean target and drop ID column
    df = raw_df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == "Yes").astype(int)

    # Train/Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Build and Fit Preprocessor
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Save processed dataframe for verification
    X_full_transformed = preprocessor.transform(X)
    X_full_transformed[TARGET_COL] = y.values
    save_processed_data(X_full_transformed, PROCESSED_DATA_PATH)

    # Save standalone preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

    # Candidate Machine Learning Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_fitted_model = None

    print("\n--- Model Training & Real Evaluation ---")
    for name, model in models.items():
        # Fit Model
        model.fit(X_train_transformed, y_train)

        # Predictions & Probabilities
        y_pred = model.predict(X_test_transformed)
        y_proba = model.predict_proba(X_test_transformed)[:, 1]

        # Calculate Genuine Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": float(np.round(acc, 4)),
            "precision": float(np.round(prec, 4)),
            "recall": float(np.round(rec, 4)),
            "f1_score": float(np.round(f1, 4)),
            "roc_auc": float(np.round(roc_auc, 4)),
            "confusion_matrix": cm,
        }

        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_fitted_model = model

    print(f"\nBest Performing Model: {best_model_name} (F1 Score: {best_f1:.4f})")

    # Build Full End-to-End Pipeline (Preprocessor + Best Model)
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_fitted_model),
    ])

    # Save Full Pipeline with joblib
    joblib.dump(full_pipeline, PIPELINE_SAVE_PATH)
    print(f"Full End-to-End Pipeline saved to {PIPELINE_SAVE_PATH}")

    # Save Metrics to JSON
    summary_output = {
        "best_model": best_model_name,
        "best_f1_score": best_f1,
        "model_evaluations": results,
    }
    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(summary_output, f, indent=2)
    print(f"Model evaluation metrics saved to {MODEL_METRICS_PATH}")

    return summary_output


if __name__ == "__main__":
    raw_data = load_raw_data()
    train_and_evaluate_pipeline(raw_data)
