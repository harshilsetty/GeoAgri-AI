"""Crop Intelligence V3: Dataset Target Capability Audit (Phase A).

Analyzes availability, missingness, physical bounds, units, coverage, and limitations
for all columns in the agricultural dataset.
Outputs: artifacts/crop_v3/dataset_capability.json
"""

import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "processed", "crop_suitability_dataset.csv")
OUTPUT_DIR = os.path.join(ROOT, "artifacts", "crop_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def audit_dataset_capability():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    n_samples, n_cols = df.shape
    print(f"Dataset Shape: {n_samples} samples x {n_cols} columns")

    fields_meta = {
        "latitude": {"units": "degrees_north", "category": "geospatial", "description": "Latitude coordinate"},
        "longitude": {"units": "degrees_east", "category": "geospatial", "description": "Longitude coordinate"},
        "state": {"units": "string", "category": "administrative", "description": "Indian State / Union Territory"},
        "year": {"units": "year", "category": "temporal", "description": "Agricultural observation year (1997-2020)"},
        "season": {"units": "categorical", "category": "agronomic_calendar", "description": "Agricultural cropping season (Kharif, Rabi, Summer, Whole Year)"},
        "crop": {"units": "string", "category": "target_crop", "description": "Observed agricultural crop class (16 classes)"},
        "soil_ph": {"units": "pH (0-14)", "category": "soil_chemistry", "description": "Electrometric soil reaction in H2O"},
        "nitrogen": {"units": "kg/ha", "category": "soil_nutrients", "description": "Available soil nitrogen"},
        "phosphorus": {"units": "kg/ha", "category": "soil_nutrients", "description": "Available soil phosphorus"},
        "potassium": {"units": "kg/ha", "category": "soil_nutrients", "description": "Available soil potassium"},
        "organic_carbon": {"units": "%", "category": "soil_health", "description": "Walkley-Black soil organic carbon"},
        "electrical_conductivity": {"units": "dS/m", "category": "soil_salinity", "description": "1:2 soil extract electrical conductivity"},
        "clay": {"units": "%", "category": "soil_texture", "description": "Mass fraction of clay particles (<0.002 mm)"},
        "sand": {"units": "%", "category": "soil_texture", "description": "Mass fraction of sand particles (>0.05 mm)"},
        "silt": {"units": "%", "category": "soil_texture", "description": "Mass fraction of silt particles (0.002-0.05 mm)"},
        "soil_texture_class": {"units": "string", "category": "soil_texture", "description": "USDA soil texture classification"},
        "temperature_mean": {"units": "°C", "category": "weather_thermal", "description": "Mean daily air temperature"},
        "temperature_min": {"units": "°C", "category": "weather_thermal", "description": "Minimum daily air temperature"},
        "temperature_max": {"units": "°C", "category": "weather_thermal", "description": "Maximum daily air temperature"},
        "humidity_mean": {"units": "%", "category": "weather_moisture", "description": "Mean relative humidity"},
        "rainfall_annual": {"units": "mm", "category": "weather_precipitation", "description": "Long-term annual precipitation sum"},
        "rainfall_season": {"units": "mm", "category": "weather_precipitation", "description": "Observed seasonal precipitation sum"},
        "rainfall_7d": {"units": "mm", "category": "weather_precipitation", "description": "Past 7-day precipitation sum"},
        "rainfall_30d": {"units": "mm", "category": "weather_precipitation", "description": "Past 30-day precipitation sum"},
        "rainfall_90d": {"units": "mm", "category": "weather_precipitation", "description": "Past 90-day precipitation sum"},
        "soil_moisture": {"units": "m³/m³", "category": "subsurface_hydrology", "description": "Volumetric rootzone soil water content"},
        "et0": {"units": "mm/day", "category": "evapotranspiration", "description": "FAO reference evapotranspiration"},
        "elevation": {"units": "meters", "category": "terrain_topography", "description": "Elevation above sea level"},
        "slope": {"units": "%", "category": "terrain_topography", "description": "Local topographic slope gradient"},
        "erosion_risk_score": {"units": "probability (0-1)", "category": "erosion_hazard", "description": "Advisory erosion risk from 8-feature XGBoost model"},
        "yield": {"units": "tonnes/hectare (t/ha)", "category": "yield_productivity", "description": "Observed crop productivity per unit cultivated area"},
    }

    audit_report = {
        "dataset_name": "Indian Agricultural Crop Suitability Dataset (IACSD-v1.0)",
        "audit_version": "v3.0",
        "sample_count": n_samples,
        "feature_count": n_cols,
        "temporal_range": {
            "start_year": int(df["year"].min()),
            "end_year": int(df["year"].max()),
            "distinct_years_count": int(df["year"].nunique()),
        },
        "geographic_coverage": {
            "states_count": int(df["state"].nunique()),
            "states": sorted(df["state"].unique().tolist()),
            "latitude_bounds": [float(df["latitude"].min()), float(df["latitude"].max())],
            "longitude_bounds": [float(df["longitude"].min()), float(df["longitude"].max())],
        },
        "crops_catalog": {
            "count": int(df["crop"].nunique()),
            "crops": sorted(df["crop"].unique().tolist()),
        },
        "field_capabilities": {},
        "missing_variables_investigation": {
            "cultivated_area": {
                "available": False,
                "notes": "Dataset records productivity (yield = production / area) in t/ha rather than raw acreage.",
            },
            "production_total_tonnes": {
                "available": False,
                "notes": "Normalized yield (t/ha) is preserved for scale-free productivity modeling across smallholders and commercial estates.",
            },
            "market_price_mandi": {
                "available": False,
                "notes": "Live mandi rates fluctuate daily; outside static physical climate training records.",
            },
            "district_names": {
                "available": False,
                "notes": "Coordinates (lat, lon) provide 250m spatial precision directly rather than administrative district centroids.",
            },
        },
        "yield_distribution_summary": {},
    }

    # Audit each column
    for col in df.columns:
        meta = fields_meta.get(col, {"units": "unknown", "category": "general", "description": col})
        series = df[col]
        is_num = pd.api.types.is_numeric_dtype(series)

        field_stat = {
            "available": True,
            "category": meta["category"],
            "units": meta["units"],
            "description": meta["description"],
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100.0), 3),
            "data_type": str(series.dtype),
        }

        if is_num:
            field_stat.update({
                "mean": round(float(series.mean()), 3),
                "std": round(float(series.std()), 3),
                "min": round(float(series.min()), 3),
                "median": round(float(series.median()), 3),
                "max": round(float(series.max()), 3),
            })
        else:
            field_stat.update({
                "unique_values_count": int(series.nunique()),
                "sample_values": series.unique().tolist()[:5],
            })

        audit_report["field_capabilities"][col] = field_stat

    # Crop-specific yield analysis
    yield_summary = {}
    for crop_name, group in df.groupby("crop"):
        yield_series = group["yield"]
        yield_summary[crop_name] = {
            "observations_count": len(yield_series),
            "mean_yield_tha": round(float(yield_series.mean()), 3),
            "median_yield_tha": round(float(yield_series.median()), 3),
            "std_yield_tha": round(float(yield_series.std()), 3),
            "p10_yield_tha": round(float(yield_series.quantile(0.10)), 3),
            "p90_yield_tha": round(float(yield_series.quantile(0.90)), 3),
            "min_yield_tha": round(float(yield_series.min()), 3),
            "max_yield_tha": round(float(yield_series.max()), 3),
        }
    audit_report["yield_distribution_summary"] = yield_summary

    output_path = os.path.join(OUTPUT_DIR, "dataset_capability.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"[OK] Phase A Dataset Capability Audit written to {output_path}")
    return audit_report


if __name__ == "__main__":
    audit_dataset_capability()
