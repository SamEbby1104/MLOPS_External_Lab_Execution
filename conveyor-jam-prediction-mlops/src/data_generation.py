import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(version="v1", records=300, seed=42):
    rng = np.random.default_rng(seed)
    festival = version == "v2"
    rated_current = rng.choice([10, 15, 20, 25], records)
    motor_current = rated_current * rng.uniform(0.55, 1.15, records)
    vibration = rng.uniform(1.0, 8.0, records)
    belt_age = rng.uniform(1, 10, records)
    throughput = rng.uniform(50, 120, records)
    dust = rng.uniform(10, 90, records)
    maintenance_days = rng.uniform(1, 180, records)
    humidity = rng.uniform(30, 90, records)

    if festival:
        throughput *= 1.30
        motor_current *= 1.08
        vibration *= 1.12
        dust = np.clip(dust * 1.10, 0, 100)

    motor_load_ratio = motor_current / rated_current
    risk_score = (
        0.30 * motor_load_ratio
        + 0.20 * vibration / 10
        + 0.15 * belt_age / 10
        + 0.15 * throughput / 150
        + 0.10 * dust / 100
        + 0.07 * maintenance_days / 180
        + 0.03 * humidity / 100
    )
    probability = 1 / (1 + np.exp(-10 * (risk_score - 0.65)))
    jam_status = rng.binomial(1, probability)

    return pd.DataFrame({
        "motor_current": motor_current,
        "rated_motor_current": rated_current,
        "vibration_level": vibration,
        "belt_age_years": belt_age,
        "parcels_per_minute": throughput,
        "dust_accumulation_index": dust,
        "days_since_maintenance": maintenance_days,
        "humidity": humidity,
        "jam_next_shift": jam_status
    })

def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = generate_dataset("v1", 300, 42)
    v2 = generate_dataset("v2", 300, 43)
    v1.to_csv(output_dir / "conveyor_v1.csv", index=False)
    v2.to_csv(output_dir / "conveyor_v2.csv", index=False)
    print("Version 1:", v1.shape, "Jam rate:", round(v1["jam_next_shift"].mean(), 3))
    print("Version 2:", v2.shape, "Jam rate:", round(v2["jam_next_shift"].mean(), 3))

if __name__ == "__main__":
    main()
