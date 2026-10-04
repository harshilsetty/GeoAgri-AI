"""Terrain Data Acquisition Module.

Interfaces with existing geo_features.py to retrieve:
1. Ground elevation (Open-Elevation API with Open-Meteo and regional fallback).
2. Terrain slope (computed from elevation deltas).
3. Evaluates baseline soil erosion risk score by querying the existing
   8-feature XGBoost erosion model (terrain_model/erosion_model.pkl).
"""

import os
import sys
import numpy as np
import joblib
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geo_features

_EROSION_MODEL = None


def _get_erosion_model():
    global _EROSION_MODEL
    if _EROSION_MODEL is None:
        candidates = [
            os.path.join(ROOT, "terrain_model", "erosion_model.pkl"),
            os.path.join(ROOT, "models", "erosion_model.pkl"),
            os.path.join(ROOT, "erosion_model.pkl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                _EROSION_MODEL = joblib.load(p)
                break
    return _EROSION_MODEL


def fetch_terrain_profile(
    lat: float,
    lon: float,
    vegetation_ratio: float = 0.35,
    rainfall_proxy: float = 35.0,
    soil_value_proxy: int = 2,
) -> Dict[str, Any]:
    """Calculates elevation, slope, and baseline erosion risk context."""
    elevation = geo_features.get_real_elevation(lat, lon)
    slope = geo_features.get_slope(lat, lon)

    model = _get_erosion_model()
    erosion_risk_score = 0.35  # default fallback
    erosion_risk_label = "MODERATE"

    if model is not None:
        n_features = getattr(model, "n_features_in_", 8)
        if n_features == 8:
            vec = np.array(
                [[slope, vegetation_ratio, elevation, rainfall_proxy, soil_value_proxy, 0.05, 0.02, 0.01]],
                dtype=np.float32,
            )
        elif n_features == 6:
            vec = np.array(
                [[slope, vegetation_ratio, elevation, 0.05, 0.02, 0.01]],
                dtype=np.float32,
            )
        else:
            vec = np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32)

        try:
            if hasattr(model, "predict_proba"):
                erosion_risk_score = float(model.predict_proba(vec)[0][1])
            else:
                erosion_risk_score = float(model.predict(vec)[0])
        except Exception:
            erosion_risk_score = 0.35

    if erosion_risk_score >= 0.70:
        erosion_risk_label = "HIGH"
    elif erosion_risk_score >= 0.30:
        erosion_risk_label = "MODERATE"
    else:
        erosion_risk_label = "LOW"

    return {
        "elevation": round(float(elevation), 1),
        "slope": round(float(slope), 2),
        "erosion_risk_score": round(float(erosion_risk_score), 4),
        "erosion_risk_label": erosion_risk_label,
        "terrain_data_quality": "HIGH" if elevation != 400.0 else "MODERATE",
    }
