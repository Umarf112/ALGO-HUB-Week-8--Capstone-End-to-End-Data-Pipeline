"""
Streamlit Reporting Dashboard & Interactive Ingestion App for AlgoHub Week 8 Capstone.
"""
import sys
import os

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    PIPELINE_SAVE_PATH,
    EDA_SUMMARY_PATH,
    MODEL_METRICS_PATH,
)
from src.utils import load_raw_data

# Page Setup
st.set_page_config(
    page_title="AlgoHub Week 8 - End-to-End Data Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ AlgoHub Week 8 Capstone: End-to-End Data Pipeline")
st.caption("IBM Telco Customer Churn - Automated EDA, Custom Sklearn Transformers, Joblib Serialization, Streamlit & FastAPI")

# Load Datasets & Artifacts Helper
@st.cache_data
def get_raw_data():
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)
    return None

@st.cache_data
def get_eda_data():
    if os.path.exists(EDA_SUMMARY_PATH):
        with open(EDA_SUMMARY_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def get_metrics_data():
    if os.path.exists(MODEL_METRICS_PATH):
        with open(MODEL_METRICS_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_resource
def get_loaded_pipeline():
    if os.path.exists(PIPELINE_SAVE_PATH):
        return joblib.load(PIPELINE_SAVE_PATH)
    return None

raw_df = get_raw_data()
eda_summary = get_eda_data()
metrics_summary = get_metrics_data()
pipeline = get_loaded_pipeline()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=70)
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Select Section",
    [
        "📌 Overview & Dataset",
        "📊 Automated EDA",
        "⚙️ Pipeline Transformation",
        "🚀 Model Metrics & API Status",
    ],
)

# API URL
API_URL = st.sidebar.text_input("FastAPI Backend URL", "http://localhost:8000")

# --- SECTION 1: OVERVIEW ---
if menu == "📌 Overview & Dataset":
    st.header("📌 Overview & Dataset Policy")
    st.markdown("""
    This project satisfies the **AlgoHub Week 8 Capstone** requirements by implementing a production-ready, reusable 
    `scikit-learn` preprocessing pipeline for the real **IBM Telco Customer Churn** dataset.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(raw_df):,}" if raw_df is not None else "0")
    with col2:
        st.metric("Total Features", f"{raw_df.shape[1]}" if raw_df is not None else "0")
    with col3:
        st.metric("Raw Space Strings (TotalCharges)", "11" if eda_summary else "N/A")
    with col4:
        st.metric("Target Variable", "Churn (Yes/No)")

    st.subheader("📄 Raw Dataset Inspection")
    if raw_df is not None:
        st.dataframe(raw_df.head(10), use_container_width=True)
    else:
        st.error("Raw dataset not found at `data/raw/Telco-Customer-Churn.csv`.")

    st.subheader("🔍 Data Types & Data Quality Issues")
    if eda_summary:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Column Data Types:**")
            st.json(eda_summary.get("data_types", {}))
        with col_right:
            st.markdown("**Data Quality Highlights:**")
            st.write("• **TotalCharges string space values:** `11` space records `' '` converted to `NaN` and imputed via median.")
            st.write("• **Class Imbalance:** 73.46% No Churn vs 26.54% Churn.")
            st.write("• **High Cardinality Categoricals:** `PaymentMethod`, `Contract`, `InternetService` mapped via custom `CategoricalEncoder`.")

# --- SECTION 2: AUTOMATED EDA ---
elif menu == "📊 Automated EDA":
    st.header("📊 Automated Exploratory Data Analysis")
    
    if eda_summary and raw_df is not None:
        # Target distribution
        t_dist = eda_summary.get("target_distribution", {})
        counts = t_dist.get("counts", {})
        
        st.subheader("1. Target Churn Distribution")
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            fig_pie = px.pie(
                values=list(counts.values()),
                names=list(counts.keys()),
                title="Churn Ratio (Yes vs No)",
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_bar:
            fig_bar = px.bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                labels={"x": "Churn Status", "y": "Customer Count"},
                title="Customer Count by Churn Status",
                color=list(counts.keys()),
                color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("2. Numerical Feature Distributions")
        num_col = st.selectbox("Select Numerical Feature", ["tenure", "MonthlyCharges", "TotalCharges"])
        
        # Clean TotalCharges for plotting
        clean_df = raw_df.copy()
        clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors="coerce")
        
        fig_hist = px.histogram(
            clean_df,
            x=num_col,
            color="Churn",
            marginal="box",
            barmode="overlay",
            title=f"Distribution of {num_col} by Churn Status",
            color_discrete_map={"No": "#3498db", "Yes": "#e74c3c"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("3. Feature Correlation Matrix")
        corr_dict = eda_summary.get("correlation_matrix", {})
        if corr_dict:
            corr_df = pd.DataFrame(corr_dict)
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap (Numeric Features & Target)",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("4. IQR Outlier Summary")
        st.json(eda_summary.get("numerical_statistics", {}))

# --- SECTION 3: PIPELINE TRANSFORMATION ---
elif menu == "⚙️ Pipeline Transformation":
    st.header("⚙️ Custom Preprocessing & Feature Engineering Pipeline")
    
    st.markdown("""
    The pipeline consists of 7 modular custom transformers inheriting from `BaseEstimator` and `TransformerMixin`:
    1. **IDColumnDropper**: Safely drops identifier columns (`customerID`).
    2. **DataTypeCoercer**: Converts string space values in `TotalCharges` to `float64`/`NaN`.
    3. **MissingValueHandler**: Median imputation for numeric and mode imputation for categorical.
    4. **OutlierCapper**: IQR Winsorization capping.
    5. **FeatureEngTransformer**: Generates `TenureYears`, `AvgMonthlyCostPerTenure`, `HasAddonsCount`, `IsLongTermContract`, `ElectronicPayment`.
    6. **CategoricalEncoder**: One-hot encodes all categorical columns.
    7. **FeatureScaler**: StandardScales numeric columns.
    """)

    st.divider()
    st.subheader("🧪 Single Customer Interactive Prediction")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure_val = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with col_b:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    with col_c:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    if st.button("Run Preprocessing & Predict Churn", type="primary"):
        input_data = pd.DataFrame([{
            "customerID": "INTERACTIVE-USER",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure_val,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])
        
        if pipeline:
            preprocessor = pipeline.named_steps["preprocessor"]
            transformed = preprocessor.transform(input_data)
            pred = pipeline.predict(input_data)[0]
            proba = pipeline.predict_proba(input_data)[0][1]

            st.success(f"**Prediction:** {'Churn (Yes)' if pred == 1 else 'No Churn (No)'}")
            st.info(f"**Churn Probability:** {proba:.4f}")
            st.write(f"**Transformed Feature Matrix Shape:** {transformed.shape}")
            with st.expander("View Transformed 51 Features"):
                st.dataframe(transformed)
        else:
            st.error("Pipeline model artifact not found.")

    st.divider()
    st.subheader("📁 Batch CSV Ingestion & Transformation")
    uploaded_file = st.file_uploader("Upload Raw CSV Dataset", type=["csv"])
    if uploaded_file and pipeline:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Raw Data:", df_uploaded.head(3))
        
        if st.button("Transform Uploaded Dataset"):
            preprocessor = pipeline.named_steps["preprocessor"]
            df_trans = preprocessor.transform(df_uploaded)
            st.success(f"Successfully transformed dataset into {df_trans.shape[1]} model-ready features!")
            st.dataframe(df_trans.head(5))
            
            csv_data = df_trans.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Processed CSV",
                csv_data,
                "processed_features.csv",
                "text/csv",
                key='download-csv'
            )

# --- SECTION 4: METRICS & API STATUS ---
elif menu == "🚀 Model Metrics & API Status":
    st.header("🚀 Machine Learning Model Performance & API Status")

    if metrics_summary:
        st.subheader("1. Genuine Model Comparison Metrics")
        st.info(f"🏆 Winning Model Selected: **{metrics_summary.get('best_model')}** (F1 Score: {metrics_summary.get('best_f1_score'):.4f})")

        evals = metrics_summary.get("model_evaluations", {})
        metrics_df = pd.DataFrame(evals).T
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in metrics_df.columns]
        display_df = metrics_df[metric_cols]
        st.dataframe(display_df.style.highlight_max(axis=0, color="#2ecc71"), use_container_width=True)

        st.subheader("2. Metrics Comparison Bar Chart")
        fig_metrics = px.bar(
            metrics_df.reset_index(),
            x="index",
            y=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            barmode="group",
            title="Evaluation Metrics Across Candidate Models",
            labels={"index": "Model Name", "value": "Metric Score", "variable": "Metric"},
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    st.divider()
    st.subheader("🌐 FastAPI Backend Connectivity Check")
    if st.button("Check Backend Connection"):
        try:
            res = requests.get(f"{API_URL}/health", timeout=3)
            if res.status_code == 200:
                st.success(f"FastAPI Backend is ONLINE at `{API_URL}`!")
                st.json(res.json())
            else:
                st.warning(f"FastAPI Backend responded with status code: {res.status_code}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI backend at `{API_URL}`: {e}")
