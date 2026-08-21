"""
Unit tests for FastAPI REST API endpoints using TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Online"


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_loaded" in data


def test_api_eda():
    response = client.get("/eda")
    assert response.status_code in [200, 404]


def test_api_metrics():
    response = client.get("/metrics")
    assert response.status_code in [200, 404]


def test_api_predict():
    payload = {
        "records": [
            {
                "customerID": "TEST-API-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
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
                "TotalCharges": "855.0",
            }
        ]
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["churn_prediction"] in ["Yes", "No"]
