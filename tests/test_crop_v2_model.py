"""Tests for Crop Intelligence Model V2.0 Artifacts & Predictions."""

import os
import sys
import joblib
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_features
import agri_inference


def test_v2_artifacts_exist():
    v2_dir = os.path.join(ROOT, "models", "crop", "v2")
    assert os.path.exists(os.path.join(v2_dir, "crop_model.pkl")), "V2 model binary missing"
    assert os.path.exists(os.path.join(v2_dir, "label_encoder.pkl")), "V2 label encoder missing"
    assert os.path.exists(os.path.join(v2_dir, "feature_metadata.json")), "V2 feature metadata missing"
    assert os.path.exists(os.path.join(v2_dir, "metrics_summary.json")), "V2 metrics summary missing"
    assert os.path.exists(os.path.join(v2_dir, "model_card.md")), "V2 model card missing"


def test_v2_model_prediction_shape():
    model, le, meta, ver = agri_inference.load_crop_artifacts("v2")
    assert ver == "v2"
    assert model is not None
    assert le is not None
    assert len(le.classes_) == 16

    # Create dummy 26-feature vector
    dummy_input = np.ones((1, 26), dtype=np.float32)
    probs = model.predict_proba(dummy_input)

    assert probs.shape == (1, 16)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)


def test_v2_inference_pipeline_execution():
    res = agri_inference.generate_crop_recommendation(
        lat=16.5,
        lon=80.6,
        season="Kharif",
        model_version="v2",
    )
    assert res["status"] == "success"
    assert res["model"]["version"] == "v2"
    assert len(res["recommendations"]) == 5
    assert "positive_drivers" in res["recommendations"][0]
    assert "limiting_factors" in res["recommendations"][0]
    assert res["recommendations"][0]["suitability_score"] > 0.0
