"""
FastAPI REST Service for Telco Customer Churn Preprocessing & Prediction Pipeline.
"""
import sys
import os

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    PIPELINE_SAVE_PATH,
    PREPROCESSOR_SAVE_PATH,
    EDA_SUMMARY_PATH,
    MODEL_METRICS_PATH,
)

app = FastAPI(
    title="AlgoHub Week 8 Capstone - Telco Data Pipeline API",
    description="Production-ready FastAPI backend for sklearn preprocessing pipeline and churn inference.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded pipeline artifacts
PIPELINE = None
PREPROCESSOR = None


@app.on_event("startup")
def load_artifacts():
    """Load serialized joblib models into memory at application startup."""
    global PIPELINE, PREPROCESSOR
    try:
        if os.path.exists(PIPELINE_SAVE_PATH):
            PIPELINE = joblib.load(PIPELINE_SAVE_PATH)
            print(f"Loaded pipeline from {PIPELINE_SAVE_PATH}")
        if os.path.exists(PREPROCESSOR_SAVE_PATH):
            PREPROCESSOR = joblib.load(PREPROCESSOR_SAVE_PATH)
            print(f"Loaded preprocessor from {PREPROCESSOR_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")


class CustomerRecord(BaseModel):
    customerID: Optional[str] = "SAMPLE-0001"
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    tenure: int = Field(..., json_schema_extra={"example": 1})
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"})
    InternetService: str = Field(..., json_schema_extra={"example": "Fiber optic"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "No"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "Yes"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "No"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., json_schema_extra={"example": 85.5})
    TotalCharges: Any = Field(..., json_schema_extra={"example": "85.5"})


class BatchRequest(BaseModel):
    records: List[CustomerRecord]


@app.get("/")
def read_root():
    return {
        "title": "AlgoHub Week 8 Capstone API",
        "status": "Online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "eda": "/eda",
            "metrics": "/metrics",
            "transform": "/transform (POST)",
            "predict": "/predict (POST)",
        },
    }


@app.get("/health")
def health_check():
    pipeline_status = PIPELINE is not None
    preprocessor_status = PREPROCESSOR is not None
    return {
        "status": "Healthy" if pipeline_status else "Degraded",
        "pipeline_loaded": pipeline_status,
        "preprocessor_loaded": preprocessor_status,
    }


@app.get("/eda")
def get_eda_report():
    """Returns the automated EDA summary JSON."""
    if not os.path.exists(EDA_SUMMARY_PATH):
        raise HTTPException(status_code=404, detail="EDA report not found. Please run pipeline runner first.")
    with open(EDA_SUMMARY_PATH, "r") as f:
        return json.load(f)


@app.get("/metrics")
def get_model_metrics():
    """Returns trained model metrics and evaluations."""
    if not os.path.exists(MODEL_METRICS_PATH):
        raise HTTPException(status_code=404, detail="Model metrics file not found. Please run pipeline runner first.")
    with open(MODEL_METRICS_PATH, "r") as f:
        return json.load(f)


@app.post("/transform")
def transform_data(payload: BatchRequest):
    """
    Ingests raw customer records and returns the transformed model-ready 51-feature matrix.
    """
    global PREPROCESSOR
    if PREPROCESSOR is None:
        if os.path.exists(PREPROCESSOR_SAVE_PATH):
            PREPROCESSOR = joblib.load(PREPROCESSOR_SAVE_PATH)
        else:
            raise HTTPException(status_code=500, detail="Preprocessor artifact not available.")

    try:
        raw_list = [rec.model_dump() for rec in payload.records]
        df_raw = pd.DataFrame(raw_list)
        df_transformed = PREPROCESSOR.transform(df_raw)
        
        return {
            "num_records": len(df_raw),
            "num_features": df_transformed.shape[1],
            "feature_matrix": df_transformed.values.tolist(),
            "feature_names": df_transformed.columns.tolist() if isinstance(df_transformed, pd.DataFrame) else [],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transformation error: {str(e)}")


@app.post("/predict")
def predict_churn(payload: BatchRequest):
    """
    Ingests raw customer records, executes the full scikit-learn pipeline,
    and returns churn prediction ('Yes'/'No'), probability, and risk level.
    """
    global PIPELINE
    if PIPELINE is None:
        if os.path.exists(PIPELINE_SAVE_PATH):
            PIPELINE = joblib.load(PIPELINE_SAVE_PATH)
        else:
            raise HTTPException(status_code=500, detail="Pipeline artifact not available.")

    try:
        raw_list = [rec.model_dump() for rec in payload.records]
        df_raw = pd.DataFrame(raw_list)

        preds = PIPELINE.predict(df_raw)
        probas = PIPELINE.predict_proba(df_raw)[:, 1]

        results = []
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            customer_id = raw_list[i].get("customerID", f"REC-{i}")
            risk = "High" if proba >= 0.65 else ("Medium" if proba >= 0.35 else "Low")
            results.append({
                "customerID": customer_id,
                "churn_prediction": "Yes" if int(pred) == 1 else "No",
                "churn_probability": float(round(proba, 4)),
                "risk_level": risk,
            })

        return {
            "total_records": len(results),
            "predictions": results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
