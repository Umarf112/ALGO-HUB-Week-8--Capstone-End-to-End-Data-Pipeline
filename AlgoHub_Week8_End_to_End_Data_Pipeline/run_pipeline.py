"""
Master Execution Script for AlgoHub Week 8 Capstone Data Pipeline.
Runs End-to-End Workflow: Data Ingestion -> Automated EDA -> Pipeline Fitting -> Model Evaluation -> Joblib Serialization -> Inference Verification.
"""
import joblib
import pandas as pd
from src.utils import load_raw_data, setup_logger
from src.automated_eda import generate_eda_report
from src.pipeline_builder import train_and_evaluate_pipeline
from src.config import PIPELINE_SAVE_PATH

logger = setup_logger("MasterPipelineRunner")

def run_end_to_end():
    logger.info("=== Starting AlgoHub Week 8 Capstone Data Pipeline Execution ===")
    
    # 1. Load Raw Data
    logger.info("Step 1: Loading raw Telco Customer Churn dataset...")
    df_raw = load_raw_data()
    logger.info(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Run Automated EDA
    logger.info("Step 2: Generating automated EDA report...")
    eda_summary = generate_eda_report(df_raw)
    logger.info(f"EDA Complete. Total space strings detected in TotalCharges: {eda_summary.get('space_strings', {})}")

    # 3. Train & Evaluate Models, Save Pipeline
    logger.info("Step 3: Building, training, and serializing full Scikit-Learn Pipeline...")
    metrics = train_and_evaluate_pipeline(df_raw)
    logger.info(f"Pipeline Serialization Complete. Winner: {metrics['best_model']}")

    # 4. Verify Loaded Pipeline Inference
    logger.info("Step 4: Verifying loaded pipeline inference on sample record...")
    loaded_pipeline = joblib.load(PIPELINE_SAVE_PATH)
    
    sample_input = pd.DataFrame([{
        "customerID": "9999-VERIFY",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": "85.5",
    }])
    
    transformed_features = loaded_pipeline.named_steps["preprocessor"].transform(sample_input)
    prediction = loaded_pipeline.predict(sample_input)[0]
    prediction_proba = loaded_pipeline.predict_proba(sample_input)[0][1]

    logger.info(f"Inference Test Output - Churn Prediction: {'Yes' if prediction == 1 else 'No'} (Probability: {prediction_proba:.4f})")
    logger.info(f"Transformed Features Shape: {transformed_features.shape}")
    
    logger.info("=== AlgoHub Week 8 Capstone End-to-End Pipeline Completed Successfully ===")

if __name__ == "__main__":
    run_end_to_end()
