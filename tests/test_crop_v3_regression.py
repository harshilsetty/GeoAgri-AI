"""Regression and Immutability Tests for Geo AI (Phase 1, 20, 21, 24).

Guarantees:
1. The 8-feature erosion XGBoost model (terrain_model/erosion_model.pkl) remains intact.
2. V1 (models/crop/v1/) and V2 (models/crop/v2/) baselines remain intact and loadable.
3. V3 (models/crop/v3/) is fully populated with model, yield regressor, metadata, and model card.
4. API backward compatibility for /api/agriculture/recommend.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_inference


def test_erosion_model_immutability():
    """Verify the 8-feature erosion model remains completely untouched."""
    model_path = os.path.join(ROOT, "terrain_model", "erosion_model.pkl")
    dataset_path = os.path.join(ROOT, "terrain_model", "erosion_dataset.csv")

    assert os.path.exists(model_path), "erosion_model.pkl missing"
    assert os.path.exists(dataset_path), "erosion_dataset.csv missing"

    model = joblib.load(model_path)
    df = pd.read_csv(dataset_path)

    expected_features = [
        "slope", "vegetation", "elevation", "rainfall",
        "soil", "boulders", "ruins", "structures"
    ]
    X = df[expected_features]
    y = df["erosion"]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Historical certified accuracy is 90.50%
    assert round(acc, 3) == 0.905, f"Erosion model performance altered! Expected 0.905, got {acc:.4f}"


def test_v1_and_v2_model_registry_preserved():
    """Verify models/crop/v1/ and models/crop/v2/ are preserved intact."""
    v1_dir = os.path.join(ROOT, "models", "crop", "v1")
    v2_dir = os.path.join(ROOT, "models", "crop", "v2")

    for v_dir in [v1_dir, v2_dir]:
        assert os.path.exists(os.path.join(v_dir, "crop_model.pkl"))
        assert os.path.exists(os.path.join(v_dir, "label_encoder.pkl"))
        assert os.path.exists(os.path.join(v_dir, "feature_metadata.json"))

    # Verify loading v2 specifically
    m2, le2, meta2, v2 = agri_inference.load_crop_artifacts("v2")
    assert v2 == "v2"
    assert m2 is not None
    assert len(le2.classes_) == 16


def test_v3_production_artifacts_exist():
    """Verify models/crop/v3/ contains all required production binaries and cards."""
    v3_dir = os.path.join(ROOT, "models", "crop", "v3")
    assert os.path.exists(v3_dir), "models/crop/v3/ directory must exist"

    required = [
        "crop_model.pkl",
        "yield_model.pkl",
        "label_encoder.pkl",
        "feature_metadata.json",
        "metrics_summary.json",
        "model_card.md",
    ]
    for fname in required:
        fpath = os.path.join(v3_dir, fname)
        assert os.path.exists(fpath), f"Missing V3 production artifact: {fname}"

    with open(os.path.join(v3_dir, "metrics_summary.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["release_status"] == "PROMOTED"
    assert "top_1_accuracy" in metrics["benchmark_metrics"]
    assert "top_5_accuracy" in metrics["benchmark_metrics"]


def test_api_recommendation_contract():
    """Verify generate_crop_recommendation() maintains complete backward compatibility."""
    rec = agri_inference.generate_crop_recommendation(15.335, 76.46, season="Kharif", top_k=5)
    
    # Existing required contract keys
    assert rec["status"] == "success"
    assert "model" in rec
    assert "location" in rec
    assert "soil" in rec
    assert "weather" in rec
    assert "terrain" in rec
    assert "erosion" in rec
    assert "recommendations" in rec
    assert "primary_recommendation" in rec
    assert "explanation" in rec
    assert "data_quality" in rec

    # V3 enhanced analytical keys
    assert "uncertainty" in rec
    assert "risk" in rec
    assert "drivers" in rec
    assert "counterfactuals" in rec
