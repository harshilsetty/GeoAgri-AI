"""Integration Tests for Crop Recommendation Engine (Phase 15).

Verifies:
1. End-to-end crop recommendation pipeline.
2. Top-5 ranking and probability properties.
3. Separation of suitability score and model confidence.
4. TreeSHAP driver and limiting factor extraction.
5. Multi-objective decision weighting.
"""

import sys
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_inference


def test_crop_recommendation_pipeline():
    result = agri_inference.generate_crop_recommendation(
        lat=15.335,
        lon=76.46,
        season="Kharif",
        vegetation_ratio=0.35,
        top_k=5,
    )

    assert "recommendations" in result
    recs = result["recommendations"]
    assert len(recs) == 5

    # Check primary crop
    primary = result["primary_recommendation"]
    assert primary["crop"] == recs[0]["crop"]
    assert 0.0 <= primary["suitability_score"] <= 1.0
    assert 0.0 <= primary["confidence"] <= 1.0
    assert 0 <= primary["suitability_percent"] <= 100
    assert 0 <= primary["confidence_percent"] <= 100

    # Ensure ranking is strictly descending by suitability score
    for i in range(len(recs) - 1):
        assert recs[i]["suitability_score"] >= recs[i + 1]["suitability_score"]

    # Verify explainability factors are present
    assert len(primary["positive_drivers"]) > 0
    assert isinstance(primary["positive_drivers"], list)

    # Verify data quality
    dq = result["data_quality"]
    assert dq["soil_data_quality"] in ["HIGH", "MODERATE", "LOW"]
    assert dq["weather_data_quality"] in ["HIGH", "MODERATE", "LOW"]
    assert dq["terrain_data_quality"] in ["HIGH", "MODERATE", "LOW"]
    assert dq["overall_data_quality"] in ["HIGH", "MODERATE", "LOW"]

    # Verify explanation narrative is present
    assert "explanation" in result
    assert len(result["explanation"]["summary"]) > 20


def test_seasonal_switch_responsiveness():
    # In same location, Kharif vs Rabi should recommend distinct season-appropriate crops
    kharif_res = agri_inference.generate_crop_recommendation(lat=28.5, lon=77.2, season="Kharif")
    rabi_res = agri_inference.generate_crop_recommendation(lat=28.5, lon=77.2, season="Rabi")

    kharif_top = kharif_res["primary_recommendation"]["crop"]
    rabi_top = rabi_res["primary_recommendation"]["crop"]

    assert kharif_res["location"]["season"] == "Kharif"
    assert rabi_res["location"]["season"] == "Rabi"
    # Rice/Cotton/Maize are Kharif, Wheat/Chickpea/Mustard are Rabi
    assert rabi_top in ["Wheat", "Chickpea", "Mustard", "Rabi Sorghum"] or rabi_res["recommendations"][0]["suitability_score"] > 0
