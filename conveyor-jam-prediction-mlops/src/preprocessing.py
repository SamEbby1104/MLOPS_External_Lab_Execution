import pandas as pd
from pathlib import Path

FEATURES = [
    "motor_current", "rated_motor_current", "vibration_level",
    "belt_age_years", "parcels_per_minute",
    "dust_accumulation_index", "days_since_maintenance", "humidity",
    "motor_load_ratio", "belt_wear_pressure_index"
]
TARGET = "jam_next_shift"

def validate_data(df):
    required = [
        "motor_current", "rated_motor_current", "vibration_level",
        "belt_age_years", "parcels_per_minute",
        "dust_accumulation_index", "days_since_maintenance",
        "humidity", "jam_next_shift"
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if (df["rated_motor_current"] <= 0).any():
        raise ValueError("Rated motor current must be positive")
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values")
    return True

def engineer_features(df):
    df = df.copy()
    df["motor_load_ratio"] = df["motor_current"] / df["rated_motor_current"]
    df["belt_wear_pressure_index"] = (
        0.30 * df["belt_age_years"] / 10
        + 0.25 * df["vibration_level"] / 10
        + 0.20 * df["dust_accumulation_index"] / 100
        + 0.15 * df["days_since_maintenance"] / 180
        + 0.10 * df["parcels_per_minute"] / 150
    )
    return df

def process_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    validate_data(df)
    df = engineer_features(df)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df

def process_all_versions():
    process_dataset("data/raw/conveyor_v1.csv", "data/processed/conveyor_v1_features.csv")
    process_dataset("data/raw/conveyor_v2.csv", "data/processed/conveyor_v2_features.csv")

if __name__ == "__main__":
    process_all_versions()
