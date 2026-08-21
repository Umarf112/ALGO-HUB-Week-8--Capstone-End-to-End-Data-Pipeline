"""
Unit tests for custom scikit-learn transformers.
"""
import numpy as np
import pandas as pd
import pytest

from src.custom_transformers import (
    IDColumnDropper,
    DataTypeCoercer,
    MissingValueHandler,
    OutlierCapper,
    FeatureEngTransformer,
    CategoricalEncoder,
    FeatureScaler,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame([
        {
            "customerID": "0001-TEST",
            "gender": "Female",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "TotalCharges": "600.0",
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
        },
        {
            "customerID": "0002-TEST",
            "gender": "Male",
            "SeniorCitizen": 1,
            "tenure": 24,
            "MonthlyCharges": 100.0,
            "TotalCharges": " ",  # Missing value as whitespace string
            "Contract": "One year",
            "PaymentMethod": "Mailed check",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
        },
    ])


def test_id_column_dropper(sample_data):
    dropper = IDColumnDropper(id_cols=["customerID"])
    transformed = dropper.fit_transform(sample_data)
    assert "customerID" not in transformed.columns
    assert "tenure" in transformed.columns


def test_data_type_coercer(sample_data):
    coercer = DataTypeCoercer(target_cols=["TotalCharges"])
    transformed = coercer.fit_transform(sample_data)
    assert pd.api.types.is_float_dtype(transformed["TotalCharges"])
    assert np.isnan(transformed["TotalCharges"].iloc[1])


def test_missing_value_handler(sample_data):
    coercer = DataTypeCoercer(target_cols=["TotalCharges"])
    data_coerced = coercer.fit_transform(sample_data)
    
    imputer = MissingValueHandler(
        numeric_cols=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_cols=["gender", "Contract"],
    )
    transformed = imputer.fit_transform(data_coerced)
    assert transformed["TotalCharges"].isnull().sum() == 0
    assert transformed["TotalCharges"].iloc[1] == 600.0


def test_outlier_capper(sample_data):
    capper = OutlierCapper(numeric_cols=["MonthlyCharges"], factor=1.5)
    transformed = capper.fit_transform(sample_data)
    assert "MonthlyCharges" in transformed.columns


def test_feature_eng_transformer(sample_data):
    fe = FeatureEngTransformer()
    transformed = fe.fit_transform(sample_data)
    assert "TenureYears" in transformed.columns
    assert "AvgMonthlyCostPerTenure" in transformed.columns
    assert "HasAddonsCount" in transformed.columns
    assert transformed["TenureYears"].iloc[0] == 1.0


def test_categorical_encoder(sample_data):
    encoder = CategoricalEncoder(categorical_cols=["Contract"])
    transformed = encoder.fit_transform(sample_data)
    assert "Contract_Month-to-month" in transformed.columns
    assert "Contract_One year" in transformed.columns


def test_feature_scaler(sample_data):
    scaler = FeatureScaler(numeric_cols=["tenure", "MonthlyCharges"])
    transformed = scaler.fit_transform(sample_data)
    assert np.isclose(transformed["tenure"].mean(), 0.0, atol=1e-5)
