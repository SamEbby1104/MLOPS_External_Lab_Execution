# Conveyor Belt Jam Prediction MLOps

End-to-end MLOps project predicting whether an automated conveyor belt is likely to jam during the next operating shift.

## Components
- Synthetic datasets V1 and festival V2
- Validation and feature engineering
- File-based feature store
- Logistic Regression and Random Forest
- MLflow experiment tracking and model registry
- FastAPI inference endpoints
- Pytest
- GitHub Actions CI

## Features
motor_current, rated_motor_current, vibration_level, belt_age_years, parcels_per_minute, dust_accumulation_index, days_since_maintenance, humidity, motor_load_ratio, belt_wear_pressure_index

Target: `jam_next_shift` (0 = Normal Operation, 1 = Jam Risk)

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run pipeline
```bash
python -m src.data_generation
python -m src.preprocessing
python -m src.train
```

## MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000

## API
```bash
uvicorn app.main:app --reload
```
Swagger UI: http://127.0.0.1:8000/docs

Health: GET `/health`

Prediction: POST `/predict`

Example request:
```json
{
  "motor_current": 18.5,
  "rated_motor_current": 20,
  "vibration_level": 7.2,
  "belt_age_years": 6.5,
  "parcels_per_minute": 145,
  "dust_accumulation_index": 80,
  "days_since_maintenance": 120,
  "humidity": 75
}
```

## Tests
```bash
pytest -v
```

GitHub Actions executes data generation, preprocessing, training and tests on push/pull request to `main`.
