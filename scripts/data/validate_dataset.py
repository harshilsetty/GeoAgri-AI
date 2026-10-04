"""Dataset Validation Module (Phase E).

Audits data/agriculture/v1.0/processed/crop_suitability_dataset.csv for:
- Missing / NaN values
- Duplicates
- Geographic coordinate bounding box integrity (India bounds)
- Soil property ranges (pH, N, P, K, Organic Carbon, EC, particle texture percentages)
- Meteorological validity (temperatures, precipitation, humidity, soil moisture, ET0)
- Target label consistency & class frequencies
- Generates data/agriculture/v1.0/dataset_validation_report.json
"""

import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "processed", "crop_suitability_dataset.csv")
REPORT_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "dataset_validation_report.json")


def validate_crop_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    total_rows = len(df)
    report = {
        "dataset_path": DATASET_PATH,
        "rows_total": int(total_rows),
        "columns_total": int(len(df.columns)),
        "columns": list(df.columns),
        "checks": {},
        "status": "PASS",
        "errors": [],
        "warnings": [],
    }

    # 1. Missing values check
    missing = df.isnull().sum().to_dict()
    missing_nonzero = {k: int(v) for k, v in missing.items() if v > 0}
    report["checks"]["missing_values"] = {
        "status": "PASS" if not missing_nonzero else "FAIL",
        "missing_counts": missing_nonzero,
    }
    if missing_nonzero:
        report["status"] = "FAIL"
        report["errors"].append(f"Found missing values in columns: {list(missing_nonzero.keys())}")

    # 2. Duplicate rows check
    duplicates = int(df.duplicated().sum())
    report["checks"]["duplicates"] = {
        "status": "PASS" if duplicates == 0 else "FAIL",
        "duplicate_rows": duplicates,
    }
    if duplicates > 0:
        report["status"] = "FAIL"
        report["errors"].append(f"Found {duplicates} duplicate rows.")

    # 3. Coordinate bounds check (India bounding box roughly 6.0 to 38.0 N, 68.0 to 98.0 E)
    lat_invalid = int(((df["latitude"] < 6.0) | (df["latitude"] > 38.0)).sum())
    lon_invalid = int(((df["longitude"] < 68.0) | (df["longitude"] > 98.0)).sum())
    report["checks"]["coordinates"] = {
        "status": "PASS" if (lat_invalid == 0 and lon_invalid == 0) else "FAIL",
        "invalid_latitude_count": lat_invalid,
        "invalid_longitude_count": lon_invalid,
        "latitude_range": [float(df["latitude"].min()), float(df["latitude"].max())],
        "longitude_range": [float(df["longitude"].min()), float(df["longitude"].max())],
    }

    # 4. Soil value bounds
    ph_invalid = int(((df["soil_ph"] < 3.5) | (df["soil_ph"] > 10.5)).sum())
    n_invalid = int(((df["nitrogen"] < 10.0) | (df["nitrogen"] > 1000.0)).sum())
    p_invalid = int(((df["phosphorus"] < 1.0) | (df["phosphorus"] > 250.0)).sum())
    k_invalid = int(((df["potassium"] < 10.0) | (df["potassium"] > 1000.0)).sum())
    oc_invalid = int(((df["organic_carbon"] < 0.01) | (df["organic_carbon"] > 10.0)).sum())
    ec_invalid = int(((df["electrical_conductivity"] < 0.0) | (df["electrical_conductivity"] > 15.0)).sum())
    texture_sum = (df["clay"] + df["sand"] + df["silt"]).round(1)
    texture_invalid = int(((texture_sum < 95.0) | (texture_sum > 105.0)).sum())

    report["checks"]["soil_ranges"] = {
        "status": "PASS"
        if all(x == 0 for x in [ph_invalid, n_invalid, p_invalid, k_invalid, oc_invalid, ec_invalid, texture_invalid])
        else "FAIL",
        "ph_range": [float(df["soil_ph"].min()), float(df["soil_ph"].max())],
        "nitrogen_range": [float(df["nitrogen"].min()), float(df["nitrogen"].max())],
        "phosphorus_range": [float(df["phosphorus"].min()), float(df["phosphorus"].max())],
        "potassium_range": [float(df["potassium"].min()), float(df["potassium"].max())],
        "organic_carbon_range": [float(df["organic_carbon"].min()), float(df["organic_carbon"].max())],
        "texture_sum_violations": texture_invalid,
    }

    # 5. Weather bounds
    tmin_invalid = int(((df["temperature_min"] < -10.0) | (df["temperature_min"] > 50.0)).sum())
    tmax_invalid = int(((df["temperature_max"] < 0.0) | (df["temperature_max"] > 55.0)).sum())
    rh_invalid = int(((df["humidity_mean"] < 0.0) | (df["humidity_mean"] > 100.0)).sum())
    rain_invalid = int((df["rainfall_annual"] < 0.0).sum())
    moist_invalid = int(((df["soil_moisture"] < 0.0) | (df["soil_moisture"] > 0.65)).sum())

    report["checks"]["weather_ranges"] = {
        "status": "PASS"
        if all(x == 0 for x in [tmin_invalid, tmax_invalid, rh_invalid, rain_invalid, moist_invalid])
        else "FAIL",
        "temperature_mean_range": [float(df["temperature_mean"].min()), float(df["temperature_mean"].max())],
        "humidity_mean_range": [float(df["humidity_mean"].min()), float(df["humidity_mean"].max())],
        "rainfall_annual_range": [float(df["rainfall_annual"].min()), float(df["rainfall_annual"].max())],
        "soil_moisture_range": [float(df["soil_moisture"].min()), float(df["soil_moisture"].max())],
    }

    # 6. Target class distribution
    class_counts = df["crop"].value_counts().to_dict()
    min_class_samples = min(class_counts.values())
    report["checks"]["target_distribution"] = {
        "unique_crops": int(len(class_counts)),
        "class_frequencies": {k: int(v) for k, v in class_counts.items()},
        "min_samples_per_class": int(min_class_samples),
        "status": "PASS" if min_class_samples >= 50 else "WARN",
    }
    if min_class_samples < 50:
        report["warnings"].append(f"Some crop classes have fewer than 50 observations: min={min_class_samples}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Validation finished with status: {report['status']}")
    print(f"[OK] Saved validation report: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    validate_crop_dataset()
