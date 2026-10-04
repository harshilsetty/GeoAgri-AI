"""Unit Tests for Geospatial Data Sources (Phase 15).

Verifies SoilGrids / Soil Health Card, Open-Meteo, and Open-Elevation / Slope modules.
"""

import sys
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.data.fetch_soil import fetch_unified_soil_profile, _resolve_agro_climatic_zone
from scripts.data.fetch_weather import fetch_live_weather
from scripts.data.fetch_terrain import fetch_terrain_profile


def test_agro_climatic_zone_resolution():
    # Hampi / Ballari (15.33, 76.46) -> Deccan / Black soil plateau
    zone_hampi = _resolve_agro_climatic_zone(15.33, 76.46)
    assert zone_hampi in ["black_cotton_vertisol", "red_sandy_alfisol"]

    # Punjab / Haryana (30.5, 75.5) -> Indo-Gangetic alluvial
    zone_punjab = _resolve_agro_climatic_zone(30.5, 75.5)
    assert zone_punjab == "gangetic_alluvial"

    # Western Rajasthan (26.5, 71.5) -> Arid desert
    zone_rajasthan = _resolve_agro_climatic_zone(26.5, 71.5)
    assert zone_rajasthan == "arid_desert"


def test_unified_soil_profile():
    soil = fetch_unified_soil_profile(15.335, 76.46)
    assert 4.0 <= soil["soil_ph"] <= 9.5
    assert soil["nitrogen"] > 0
    assert soil["phosphorus"] > 0
    assert soil["potassium"] > 0
    assert soil["organic_carbon"] > 0
    assert soil["clay"] + soil["sand"] + soil["silt"] == pytest.approx(100.0, abs=2.0)
    assert soil["soil_data_quality"] in ["HIGH", "MODERATE", "LOW"]


def test_weather_retrieval_and_aggregation():
    weather = fetch_live_weather(15.335, 76.46)
    assert -10.0 <= weather["temperature_mean"] <= 55.0
    assert 0.0 <= weather["humidity_mean"] <= 100.0
    assert weather["rainfall_7d"] >= 0.0
    assert weather["rainfall_30d"] >= 0.0
    assert weather["forecast_rainfall_7d"] >= 0.0
    assert 0.0 <= weather["soil_moisture"] <= 0.60
    assert weather["weather_data_quality"] in ["HIGH", "MODERATE", "LOW"]


def test_terrain_and_erosion_context():
    terrain = fetch_terrain_profile(15.335, 76.46, vegetation_ratio=0.35)
    assert terrain["elevation"] >= 0.0
    assert terrain["slope"] >= 0.0
    assert 0.0 <= terrain["erosion_risk_score"] <= 1.0
    assert terrain["erosion_risk_label"] in ["LOW", "MODERATE", "HIGH"]
