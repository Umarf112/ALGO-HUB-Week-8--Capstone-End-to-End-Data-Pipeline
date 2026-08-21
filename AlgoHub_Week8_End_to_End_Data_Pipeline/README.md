# AlgoHub Week 8 Capstone: End-to-End Data Pipeline

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Scikit--Learn-1.7%2B-orange)
![API](https://img.shields.io/badge/FastAPI-0.121%2B-green)
![Dashboard](https://img.shields.io/badge/Streamlit-1.62%2B-red)
![Tests](https://img.shields.io/badge/Pytest-15%2F15%20Passed-brightgreen)

An end-to-end, production-grade data preprocessing and machine learning pipeline built on the **IBM Telco Customer Churn** dataset for the **AlgoHub Data Analysis Internship Capstone**.

---

## 📌 Executive Summary

Clean, well-engineered data is the foundation of production AI products. This project constructs an automated, modular, leak-free Scikit-Learn data pipeline featuring custom transformers (`BaseEstimator`, `TransformerMixin`), automated EDA, model benchmarking, `joblib` serialization, a `FastAPI` REST backend, and a `Streamlit` interactive reporting dashboard.

---

## 📂 Repository Structure

```
AlgoHub_Week8_End_to_End_Data_Pipeline/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv             # Raw IBM Telco Churn Dataset (7,043 rows x 21 cols)
│   └── processed/
│       └── Telco-Customer-Churn-Processed.csv   # Model-ready feature matrix (7,043 rows x 52 cols)
├── models/
│   ├── pipeline.joblib                           # Full serialized end-to-end Pipeline (Preprocessor + Model)
│   ├── preprocessor.joblib                       # Standalone Preprocessor Pipeline
│   ├── eda_summary.json                          # Automated EDA report
│   └── model_metrics.json                        # Model evaluation & benchmark metrics
├── src/
│   ├── __init__.py
│   ├── config.py                                 # Centralized configuration & feature schema
│   ├── custom_transformers.py                    # Modular Sklearn Transformers (BaseEstimator + TransformerMixin)
│   ├── pipeline_builder.py                       # Pipeline assembly, model fitting & joblib serialization
│   ├── automated_eda.py                          # Automated EDA generation engine
│   └── utils.py                                  # Data I/O, directory creation, logging utilities
├── api/
│   ├── __init__.py
│   └── main.py                                   # FastAPI REST Backend (/health, /eda, /transform, /predict)
├── app/
│   └── streamlit_app.py                          # Streamlit Interactive Reporting & Ingestion Dashboard
├── notebooks/
│   └── week8_capstone_pipeline.ipynb             # Interactive demonstration Jupyter notebook
├── tests/
│   ├── __init__.py
│   ├── test_transformers.py                      # Unit tests for custom sklearn transformers
│   ├── test_pipeline.py                         # Unit tests for pipeline fitting & serialization
│   └── test_api.py                               # Unit tests for FastAPI REST endpoints
├── README.md                                     # Official Capstone Documentation & Setup Guide
├── requirements.txt                              # Project dependencies
├── run_pipeline.py                               # Master end-to-end execution runner
```

---

## 📊 Dataset & Preprocessing Decisions

### Raw Dataset Details
* **Source**: Real IBM Telco Customer Churn Dataset (`data/raw/Telco-Customer-Churn.csv`)
* **Shape**: 7,043 rows x 21 columns
* **Target Column**: `Churn` (`Yes`: 26.54%, `No`: 73.46%)

### Real Data Quality Issues Identified
1. **Identifier Columns**: `customerID` (unique string) removed before modeling using custom `IDColumnDropper`.
2. **Whitespace Missing Values**: `TotalCharges` contained 11 whitespace strings (`' '`). Coerced to `float64` / `NaN` using custom `DataTypeCoercer` and imputed via median with `MissingValueHandler`.
3. **Categorical Features**: High cardinality features (`PaymentMethod`, `Contract`, `InternetService`, etc.) encoded into binary indicators via custom `CategoricalEncoder` (One-Hot Encoding with feature name preservation).
4. **Outlier Capping**: IQR-based Winsorization applied to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `OutlierCapper`.
5. **Feature Engineering**: Created domain-specific features using `FeatureEngTransformer`:
   - `TenureYears` = `tenure / 12.0`
   - `AvgMonthlyCostPerTenure` = `TotalCharges / (tenure + 1.0)`
   - `HasAddonsCount` = sum of active security/support/backup internet add-ons
   - `IsLongTermContract` = 1 if contract is 1 or 2 years, else 0
   - `ElectronicPayment` = 1 if payment method is Electronic check, else 0

---

## ⚙️ Custom Scikit-Learn Transformers

All transformers in `src/custom_transformers.py` inherit from `sklearn.base.BaseEstimator` and `TransformerMixin`:

```python
class IDColumnDropper(BaseEstimator, TransformerMixin): ...
class DataTypeCoercer(BaseEstimator, TransformerMixin): ...
class MissingValueHandler(BaseEstimator, TransformerMixin): ...
class OutlierCapper(BaseEstimator, TransformerMixin): ...
class FeatureEngTransformer(BaseEstimator, TransformerMixin): ...
class CategoricalEncoder(BaseEstimator, TransformerMixin): ...
class FeatureScaler(BaseEstimator, TransformerMixin): ...
```

---

## 🏆 Model Benchmarking & Genuine Results

Models were evaluated using an 80/20 stratified train/test split.

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8105** | **0.6814** | **0.5374** | **0.6009** | **0.8471** | 🏆 **Selected Winner** |
| **Gradient Boosting** | 0.8048 | 0.6737 | 0.5134 | 0.5827 | 0.8432 | Runner-Up |
| **Random Forest** | 0.7913 | 0.6342 | 0.5053 | 0.5625 | 0.8235 | Baseline |

---

## 🚀 Quickstart & Execution Guide

### 1. Installation
```bash
# Clone the repository and navigate into the project
cd AlgoHub_Week8_End_to_End_Data_Pipeline

# Install dependencies
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Executes automated EDA, fits the preprocessing and model pipeline, serializes artifacts with `joblib`, and verifies sample inference:
```bash
python run_pipeline.py
```

### 3. Run Pytest Suite
Executes all 15 unit tests:
```bash
python -m pytest tests/ -v
```

### 4. Launch FastAPI REST Service
```bash
uvicorn api.main:app --reload --port 8000
```
* **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
* **Endpoints**:
  - `GET /`: API Overview & Metadata
  - `GET /health`: System Health & Artifact Check
  - `GET /eda`: Automated EDA Summary JSON
  - `GET /metrics`: Model Benchmark Metrics JSON
  - `POST /transform`: Raw records to transformed 51-feature matrix
  - `POST /predict`: Churn prediction, probability, and risk classification

### 5. Launch Streamlit Reporting Dashboard
```bash
streamlit run app/streamlit_app.py
```
Provides 4 interactive pages:
1. **Overview & Dataset**: Raw dataset preview & data quality highlights.
2. **Automated EDA**: Plotly interactive charts (Churn ratio, feature distributions, correlation matrix).
3. **Pipeline Transformation**: Single customer interactive predictor & batch CSV transformer with download button.
4. **Model Metrics & API Status**: Model comparisons & live FastAPI connection health check.

---

## 📜 License & Acknowledgments

* **Handbook**: AlgoHub Software House Official Internship Handbook (Week 8 Capstone)
* **Dataset**: Real IBM Telco Customer Churn Dataset
