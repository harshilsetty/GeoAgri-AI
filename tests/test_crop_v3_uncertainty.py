"""Tests for Recommendation Uncertainty, Decision Margin, and Stability (Phase L)."""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_inference


def test_uncertainty_fields_in_recommendation():
    rec = agri_inference.generate_crop_recommendation(15.335, 76.46, season="Kharif", top_k=5)
    assert "uncertainty" in rec, "Recommendation must contain 'uncertainty' block"
    u = rec["uncertainty"]

    assert "decision_margin" in u
    assert "ranking_stability" in u
    assert "recommendation_uncertainty" in u
    assert "interpretation" in u

    # Check bounds
    assert isinstance(u["decision_margin"], (int, float))
    assert u["decision_margin"] >= 0.0
    assert 0.0 <= u["recommendation_uncertainty"] <= 1.0
    assert u["ranking_stability"] in [
        "High Stability",
        "Moderate Stability",
        "Close Decision (Multiple Crops Viable)",
    ]
    assert len(u["interpretation"]) > 10


def test_risk_breakdown_per_candidate():
    rec = agri_inference.generate_crop_recommendation(15.335, 76.46, season="Kharif", top_k=5)
    for crop in rec["recommendations"]:
        assert "risk" in crop
        assert crop["risk"] in ["Low", "Moderate", "High"]
        assert "risk_breakdown" in crop
        rb = crop["risk_breakdown"]
        assert "weather_risk" in rb
        assert "terrain_risk" in rb
        assert "erosion_risk" in rb
        assert rb["weather_risk"] in ["Low", "Moderate", "High"]
        assert rb["terrain_risk"] in ["Low", "Moderate", "High"]
        assert rb["erosion_risk"] in ["Low", "Moderate", "High"]
