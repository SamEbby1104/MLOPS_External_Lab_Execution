import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.preprocessing import FEATURES

app = FastAPI(
    title="Conveyor Jam Prediction API",
    description="Predicts conveyor belt jam risk for the next operating shift",
    version="1.0.0"
)
MODEL_PATH = Path("models/best_model.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class SensorInput(BaseModel):
    motor_current: float = Field(gt=0)
    rated_motor_current: float = Field(gt=0)
    vibration_level: float = Field(ge=0)
    belt_age_years: float = Field(ge=0)
    parcels_per_minute: float = Field(ge=0)
    dust_accumulation_index: float = Field(ge=0, le=100)
    days_since_maintenance: float = Field(ge=0)
    humidity: float = Field(ge=0, le=100)

def create_features(data):
    load_ratio = data.motor_current / data.rated_motor_current
    wear_index = (
        0.30 * data.belt_age_years / 10
        + 0.25 * data.vibration_level / 10
        + 0.20 * data.dust_accumulation_index / 100
        + 0.15 * data.days_since_maintenance / 180
        + 0.10 * data.parcels_per_minute / 150
    )
    values = {
        "motor_current": data.motor_current,
        "rated_motor_current": data.rated_motor_current,
        "vibration_level": data.vibration_level,
        "belt_age_years": data.belt_age_years,
        "parcels_per_minute": data.parcels_per_minute,
        "dust_accumulation_index": data.dust_accumulation_index,
        "days_since_maintenance": data.days_since_maintenance,
        "humidity": data.humidity,
        "motor_load_ratio": load_ratio,
        "belt_wear_pressure_index": wear_index
    }
    return pd.DataFrame([values])[FEATURES]

@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "model_not_loaded",
            "model_loaded": model is not None}

@app.post("/predict")
def predict(data: SensorInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    features = create_features(data)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1]) if hasattr(model, "predict_proba") else None
    return {
        "prediction": "Jam Risk" if prediction == 1 else "Normal Operation",
        "jam_prediction": prediction,
        "risk_probability": probability,
        "motor_load_ratio": round(data.motor_current / data.rated_motor_current, 3),
        "belt_wear_pressure_index": round(float(features["belt_wear_pressure_index"].iloc[0]), 3)
    }
