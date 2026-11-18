from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction_execution():
    data = {
        "motor_current": 10, "rated_motor_current": 20,
        "vibration_level": 2, "belt_age_years": 2,
        "parcels_per_minute": 50, "dust_accumulation_index": 20,
        "days_since_maintenance": 10, "humidity": 50
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in ["Jam Risk", "Normal Operation"]
    assert "risk_probability" in result
    assert "motor_load_ratio" in result

def test_invalid_input():
    data = {
        "motor_current": -10, "rated_motor_current": 20,
        "vibration_level": 2, "belt_age_years": 2,
        "parcels_per_minute": 50, "dust_accumulation_index": 20,
        "days_since_maintenance": 10, "humidity": 50
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422
