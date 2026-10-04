"""Tests for Agronomic Knowledge Layer and Physiological Compatibility Engine (Phase E, F, H, K)."""

import json
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train_crop_v3 import PhysiologicalCompatibilityEngine


def test_crop_knowledge_base_integrity():
    kb_path = os.path.join(ROOT, "data", "agriculture", "knowledge", "v1", "crop_knowledge_base.json")
    assert os.path.exists(kb_path), f"Knowledge base missing at {kb_path}"

    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    assert "crops" in kb_data
    crops = kb_data["crops"]
    assert len(crops) == 16, f"Expected 16 crops, found {len(crops)}"

    required_fields = [
        "preferred_ph_min", "preferred_ph_max",
        "temperature_min", "temperature_max",
        "rainfall_min", "rainfall_max",
        "soil_moisture_range", "season", "soil_texture",
        "water_requirement", "erosion_tolerance",
        "source", "source_url", "reference", "confidence"
    ]

    for crop_name, crop_info in crops.items():
        for field in required_fields:
            assert field in crop_info, f"Crop {crop_name} missing required field '{field}'"

        # Verify physical bounds
        assert crop_info["preferred_ph_min"] <= crop_info["preferred_ph_max"]
        assert crop_info["temperature_min"] <= crop_info["temperature_max"]
        assert crop_info["rainfall_min"] <= crop_info["rainfall_max"]
        assert 0.0 < crop_info["confidence"] <= 1.0
        assert len(crop_info["source_url"]) > 10
        assert len(crop_info["reference"]) > 10


def test_physiological_compatibility_calculation():
    engine = PhysiologicalCompatibilityEngine()

    # Optimal environment for Rice (pH ~6.5, high rainfall ~1500, warm temp ~28, flat slope ~2)
    optimal_rice_env = {
        "soil_ph": 6.5,
        "soil_texture_class": "Clay",
        "temperature_mean": 28.0,
        "rainfall_season": 1500.0,
        "soil_moisture": 0.45,
        "season": "Kharif",
        "slope": 2.0,
        "erosion_risk_score": 0.1,
    }

    comp = engine.evaluate_crop_compatibility(optimal_rice_env, "Rice")
    assert 0.80 <= comp["s_composite"] <= 1.0, f"Rice should have high compatibility, got {comp['s_composite']}"
    assert comp["s_soil"] >= 0.90
    assert comp["s_climate"] >= 0.90
    assert comp["s_season"] == 1.0

    # Incompatible environment for Rice (arid desert: pH 8.5, rain 150mm, sandy, off-season)
    arid_env = {
        "soil_ph": 8.8,
        "soil_texture_class": "Sand",
        "temperature_mean": 42.0,
        "rainfall_season": 100.0,
        "soil_moisture": 0.08,
        "season": "Rabi",
        "slope": 25.0,
        "erosion_risk_score": 0.85,
    }

    arid_comp = engine.evaluate_crop_compatibility(arid_env, "Rice")
    assert arid_comp["s_composite"] <= 0.55, f"Rice should be heavily penalized in arid env, got {arid_comp['s_composite']}"
    assert arid_comp["s_climate"] < 0.30
    assert arid_comp["s_soil"] < 0.60


def test_strict_seasonal_penalty():
    engine = PhysiologicalCompatibilityEngine()
    env_kharif = {
        "soil_ph": 6.5,
        "soil_texture_class": "Loam",
        "temperature_mean": 28.0,
        "rainfall_season": 600.0,
        "soil_moisture": 0.25,
        "season": "Kharif",
        "slope": 3.0,
        "erosion_risk_score": 0.2,
    }

    # Wheat is strictly Rabi
    wheat_comp = engine.evaluate_crop_compatibility(env_kharif, "Wheat")
    assert wheat_comp["s_season"] <= 0.10, "Wheat must receive severe seasonal penalty in Kharif"
