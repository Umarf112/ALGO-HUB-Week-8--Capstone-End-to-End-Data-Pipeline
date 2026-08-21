"""
Unit tests for full pipeline execution, fitting, and joblib serialization.
"""
import os
import joblib
import pandas as pd
import pytest
from src.config import RAW_DATA_PATH, PIPELINE_SAVE_PATH, PREPROCESSOR_SAVE_PATH
from src.pipeline_builder import build_preprocessor, train_and_evaluate_pipeline
from src.utils import load_raw_data


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    assert len(preprocessor.steps) == 7


def test_full_pipeline_train_and_serialize():
    if not os.path.exists(RAW_DATA_PATH):
        pytest.skip("Raw dataset not found.")
    
    df_raw = load_raw_data().head(100)
    results = train_and_evaluate_pipeline(df_raw)
    
    assert "best_model" in results
    assert os.path.exists(PIPELINE_SAVE_PATH)
    assert os.path.exists(PREPROCESSOR_SAVE_PATH)


def test_loaded_pipeline_inference():
    if not os.path.exists(PIPELINE_SAVE_PATH):
        pytest.skip("Pipeline artifact not built.")
        
    pipeline = joblib.load(PIPELINE_SAVE_PATH)
    
    sample_df = pd.DataFrame([{
        "customerID": "9999-TEST",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 45.0,
        "TotalCharges": "225.0",
    }])
    
    preds = pipeline.predict(sample_df)
    probas = pipeline.predict_proba(sample_df)
    
    assert len(preds) == 1
    assert preds[0] in [0, 1]
    assert probas.shape == (1, 2)
