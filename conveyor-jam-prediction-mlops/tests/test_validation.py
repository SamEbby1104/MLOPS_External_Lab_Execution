import pytest
import pandas as pd
from src.preprocessing import validate_data

def sample_df():
    return pd.DataFrame({
        "motor_current": [10], "rated_motor_current": [20],
        "vibration_level": [2], "belt_age_years": [2],
        "parcels_per_minute": [50], "dust_accumulation_index": [20],
        "days_since_maintenance": [10], "humidity": [50],
        "jam_next_shift": [0]
    })

def test_valid_sensor_data():
    assert validate_data(sample_df()) is True

def test_invalid_rated_current():
    df = sample_df()
    df["rated_motor_current"] = 0
    with pytest.raises(ValueError):
        validate_data(df)

def test_missing_column():
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame({"motor_current": [10]}))

def test_missing_values():
    df = sample_df()
    df.loc[0, "humidity"] = None
    with pytest.raises(ValueError):
        validate_data(df)
