"""Unit Tests for Feature Engineering (Phase 15).

Verifies derived agronomic metrics, bounds, and encoding consistency.
"""

import sys
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_features


def test_soil_fertility_index_bounds():
    sfi_low = agri_features.calculate_soil_fertility_index(
        nitrogen=60.0, phosphorus=6.0, potassium=60.0, organic_carbon=0.2, ph=5.0
    )
    sfi_high = agri_features.calculate_soil_fertility_index(
        nitrogen=350.0, phosphorus=35.0, potassium=320.0, organic_carbon=1.2, ph=6.8
    )

    assert 0.0 <= sfi_low <= 100.0
    assert 0.0 <= sfi_high <= 100.0
    assert sfi_high > sfi_low, "Fertile soil must produce higher index than nutrient-depleted soil"


def test_water_stress_index():
    # Low stress: high rainfall/moisture, low ET0
    wsi_low = agri_features.calculate_water_stress_index(et0=3.0, rainfall_30d=180.0, soil_moisture=0.35)
    # High stress: high ET0, low rain/moisture
    wsi_high = agri_features.calculate_water_stress_index(et0=7.5, rainfall_30d=5.0, soil_moisture=0.10)

    assert wsi_low < wsi_high, "Arid/low moisture must have higher water stress than moist ground"
    assert wsi_low >= 0.1
    assert wsi_high <= 5.0


def test_engineer_features_row_schema():
    raw_sample = {
        "soil_ph": 7.2,
        "nitrogen": 210.0,
        "phosphorus": 18.0,
        "potassium": 240.0,
        "organic_carbon": 0.65,
        "electrical_conductivity": 0.25,
        "clay": 30.0,
        "sand": 40.0,
        "silt": 30.0,
        "temperature_mean": 27.0,
        "temperature_min": 21.0,
        "temperature_max": 33.0,
        "humidity_mean": 65.0,
        "rainfall_annual": 900.0,
        "rainfall_season": 600.0,
        "rainfall_30d": 150.0,
        "rainfall_90d": 450.0,
        "soil_moisture": 0.25,
        "et0": 4.8,
        "elevation": 400.0,
        "slope": 6.5,
        "erosion_risk_score": 0.25,
        "season": "Kharif",
        "soil_texture_class": "Clay Loam",
    }

    features = agri_features.engineer_features_row(raw_sample)
    assert len(features) == len(agri_features.FEATURE_COLUMNS)
    for col in agri_features.FEATURE_COLUMNS:
        assert col in features
        assert isinstance(features[col], (int, float))
