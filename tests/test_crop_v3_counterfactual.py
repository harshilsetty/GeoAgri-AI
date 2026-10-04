"""Tests for Model-Supported Counterfactual Perturbation Analysis (Phase M)."""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_inference


def test_counterfactual_scenarios_generated():
    rec = agri_inference.generate_crop_recommendation(15.335, 76.46, season="Kharif", top_k=5)
    assert "counterfactuals" in rec, "Recommendation must contain 'counterfactuals'"
    cfs = rec["counterfactuals"]
    assert len(cfs) >= 4, f"Expected at least 4 counterfactual scenarios, got {len(cfs)}"

    required_keys = ["scenario", "top_candidate", "simulated_suitability_percent", "is_rank_shift", "insight"]
    for cf in cfs:
        for k in required_keys:
            assert k in cf, f"Counterfactual missing key '{k}'"
        assert 0 <= cf["simulated_suitability_percent"] <= 100
        assert isinstance(cf["is_rank_shift"], bool)
        assert len(cf["insight"]) > 10


def test_evaluate_counterfactuals_function():
    crop_model, le, meta, _ = agri_inference.load_crop_artifacts("v3")
    sample_env = {
        "soil_ph": 7.2,
        "nitrogen": 220.0,
        "phosphorus": 22.0,
        "potassium": 250.0,
        "organic_carbon": 0.55,
        "electrical_conductivity": 0.25,
        "clay": 30.0,
        "sand": 40.0,
        "silt": 30.0,
        "temperature_mean": 27.0,
        "temperature_range": 10.0,
        "humidity_mean": 65.0,
        "rainfall_season": 650.0,
        "rainfall_30d": 120.0,
        "rainfall_90d": 350.0,
        "soil_moisture": 0.28,
        "et0": 4.5,
        "elevation": 350.0,
        "slope": 4.0,
        "erosion_risk_score": 0.25,
        "soil_fertility_index": 72.0,
        "water_stress_index": 0.35,
        "rainfall_deviation": 5.0,
        "soil_moisture_index": 70.0,
        "season_code": 0,
        "texture_code": 2,
    }

    cfs = agri_inference.evaluate_counterfactuals(
        crop_model=crop_model,
        label_encoder=le,
        base_features_dict=sample_env,
        base_top_crop="Groundnut",
    )
    assert len(cfs) >= 4
    scenarios = [c["scenario"] for c in cfs]
    assert any("Drier" in s for s in scenarios)
    assert any("Wetter" in s for s in scenarios)
