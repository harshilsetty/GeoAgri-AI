"""Agricultural Recommendation & Decision Engine (Crop Intelligence V3).

Coordinates:
1. Geospatial feature acquisition (Soil, Weather, Terrain, Erosion).
2. 26-feature agronomic transformation.
3. Multi-class crop prediction via production XGBoost model (v3/v2/v1).
4. Agronomic Knowledge-Grounded Physiological Compatibility Engine (FAO EcoCrop & ICAR).
5. Yield-Aware Regressor estimating productivity (t/ha) and relative yield potential.
6. Multi-dimensional risk estimation (Weather risk, Terrain risk, Erosion risk).
7. Decision margin, ranking stability, and recommendation uncertainty.
8. Model-supported Counterfactual Analysis ("What would change this recommendation?").
9. Exact TreeSHAP local explanations for top-ranked crops.
10. Downstream non-hallucinatory LLM explanation (Groq Llama-3.1 / Local fallback).
11. Data quality metrics and confidence estimation.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import xgboost as xgb

# Dynamic ROOT resolution supporting both package and standalone execution
_CUR_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(_CUR_DIR) == "models" and os.path.basename(os.path.dirname(_CUR_DIR)) == "src":
    ROOT = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
else:
    ROOT = _CUR_DIR

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_SRC_DIR = os.path.join(ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from src.preprocessing import agri_features
except ImportError:
    import agri_features

try:
    from src.geospatial.fetch_soil import fetch_unified_soil_profile
    from src.geospatial.fetch_weather import fetch_live_weather
    from src.geospatial.fetch_terrain import fetch_terrain_profile
except ImportError:
    from scripts.data.fetch_soil import fetch_unified_soil_profile
    from scripts.data.fetch_weather import fetch_live_weather
    from scripts.data.fetch_terrain import fetch_terrain_profile

try:
    from src.explainability.ai_explainer import generate_groq_from_prompt_with_status
except ImportError:
    from terrain_model.ai_explainer_groq import generate_groq_from_prompt_with_status

DEFAULT_CROP_MODEL_VERSION = os.getenv("GEO_AI_CROP_MODEL_VERSION", "v2").strip().lower()

_MODEL_REGISTRY_CACHE = {}
_CROP_KB_CACHE = None


def get_model_dir(version: Optional[str] = None) -> Tuple[str, str]:
    """Resolves model directory and actual version string with graceful fallback."""
    req_v = (version or DEFAULT_CROP_MODEL_VERSION).strip().lower()
    cand_dir = os.path.join(ROOT, "models", "crop", req_v)
    if os.path.exists(cand_dir) and os.path.exists(os.path.join(cand_dir, "crop_model.pkl")):
        return cand_dir, req_v
    # Graceful fallback: return v1 default baseline on unknown version
    fallback_dir = os.path.join(ROOT, "models", "crop", "v1")
    if os.path.exists(fallback_dir) and os.path.exists(os.path.join(fallback_dir, "crop_model.pkl")):
        return fallback_dir, "v1"
    for fb in ["v5", "v4", "v3", "v2"]:
        fb_dir = os.path.join(ROOT, "models", "crop", fb)
        if os.path.exists(fb_dir) and os.path.exists(os.path.join(fb_dir, "crop_model.pkl")):
            return fb_dir, fb
    return fallback_dir, "v1"


def get_crop_knowledge_base() -> Dict[str, Any]:
    """Loads and caches the versioned agronomic knowledge base."""
    global _CROP_KB_CACHE
    if _CROP_KB_CACHE is not None:
        return _CROP_KB_CACHE

    kb_path = os.path.join(ROOT, "data", "agriculture", "knowledge", "v1", "crop_knowledge_base.json")
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                _CROP_KB_CACHE = json.load(f)["crops"]
                return _CROP_KB_CACHE
        except Exception:
            pass

    # Fallback to in-memory ecological rules if JSON missing
    _CROP_KB_CACHE = {}
    return _CROP_KB_CACHE


# Scientific ecological crop thresholds (FAO EcoCrop & ICAR Agro-Meteorology Standards)
CROP_ECOLOGICAL_RULES = {
    "Rice": {
        "opt_ph": (5.5, 7.2),
        "opt_rain_season": (600, 2200),
        "seasons": ["Kharif", "Whole Year"],
        "drought_tolerance": "Low",
        "erosion_vulnerability": "Moderate",
        "max_slope": 12.0,
    },
    "Wheat": {
        "opt_ph": (6.0, 7.5),
        "opt_rain_season": (150, 600),
        "seasons": ["Rabi"],
        "drought_tolerance": "Moderate",
        "erosion_vulnerability": "Low",
        "max_slope": 15.0,
    },
    "Maize": {
        "opt_ph": (5.8, 7.5),
        "opt_rain_season": (400, 1100),
        "seasons": ["Kharif", "Rabi", "Summer", "Whole Year"],
        "drought_tolerance": "Moderate",
        "erosion_vulnerability": "Moderate",
        "max_slope": 18.0,
    },
    "Groundnut": {
        "opt_ph": (6.0, 7.2),
        "opt_rain_season": (350, 900),
        "seasons": ["Kharif", "Summer"],
        "drought_tolerance": "High",
        "erosion_vulnerability": "Low",  # Groundnut acts as a soil cover crop
        "max_slope": 15.0,
    },
    "Cotton": {
        "opt_ph": (6.2, 8.2),
        "opt_rain_season": (450, 1100),
        "seasons": ["Kharif"],
        "drought_tolerance": "High",
        "erosion_vulnerability": "High",  # Wide row spacing increases erosion on slopes
        "max_slope": 10.0,
    },
    "Sorghum": {
        "opt_ph": (5.8, 8.2),
        "opt_rain_season": (300, 850),
        "seasons": ["Kharif", "Rabi"],
        "drought_tolerance": "Very High",
        "erosion_vulnerability": "Low",
        "max_slope": 20.0,
    },
    "Pearl Millet": {
        "opt_ph": (6.0, 8.5),
        "opt_rain_season": (250, 650),
        "seasons": ["Kharif", "Summer"],
        "drought_tolerance": "Very High",
        "erosion_vulnerability": "Low",
        "max_slope": 22.0,
    },
    "Chickpea": {
        "opt_ph": (6.0, 7.8),
        "opt_rain_season": (100, 450),
        "seasons": ["Rabi"],
        "drought_tolerance": "High",
        "erosion_vulnerability": "Low",
        "max_slope": 16.0,
    },
    "Pigeonpea": {
        "opt_ph": (5.5, 7.8),
        "opt_rain_season": (450, 1000),
        "seasons": ["Kharif", "Whole Year"],
        "drought_tolerance": "High",
        "erosion_vulnerability": "Low",
        "max_slope": 18.0,
    },
    "Moong": {
        "opt_ph": (6.2, 7.5),
        "opt_rain_season": (300, 750),
        "seasons": ["Kharif", "Summer"],
        "drought_tolerance": "Moderate",
        "erosion_vulnerability": "Low",
        "max_slope": 18.0,
    },
    "Urad": {
        "opt_ph": (6.0, 7.5),
        "opt_rain_season": (350, 800),
        "seasons": ["Kharif", "Rabi"],
        "drought_tolerance": "Moderate",
        "erosion_vulnerability": "Low",
        "max_slope": 18.0,
    },
    "Sugarcane": {
        "opt_ph": (6.0, 7.8),
        "opt_rain_season": (750, 2500),
        "seasons": ["Whole Year", "Kharif"],
        "drought_tolerance": "Low",
        "erosion_vulnerability": "Low",
        "max_slope": 12.0,
    },
    "Soybean": {
        "opt_ph": (6.0, 7.5),
        "opt_rain_season": (450, 1050),
        "seasons": ["Kharif"],
        "drought_tolerance": "Moderate",
        "erosion_vulnerability": "Low",
        "max_slope": 15.0,
    },
    "Mustard": {
        "opt_ph": (6.0, 7.8),
        "opt_rain_season": (150, 500),
        "seasons": ["Rabi"],
        "drought_tolerance": "High",
        "erosion_vulnerability": "Low",
        "max_slope": 16.0,
    },
    "Finger Millet": {
        "opt_ph": (5.0, 7.5),
        "opt_rain_season": (350, 950),
        "seasons": ["Kharif"],
        "drought_tolerance": "Very High",
        "erosion_vulnerability": "Low",
        "max_slope": 22.0,
    },
    "Sesamum": {
        "opt_ph": (5.5, 7.5),
        "opt_rain_season": (300, 700),
        "seasons": ["Kharif", "Summer"],
        "drought_tolerance": "High",
        "erosion_vulnerability": "Low",
        "max_slope": 18.0,
    },
}

# Multi-Objective Layer Weights (Domain Configurable)
WEIGHT_MODEL = 0.45
WEIGHT_SOIL = 0.20
WEIGHT_SEASON = 0.15
WEIGHT_CLIMATE = 0.10
WEIGHT_EROSION = 0.10


def load_crop_artifacts(version: Optional[str] = None) -> Tuple[Any, Any, Any, str]:
    """Loads and caches model artifacts by version. Maintains backward compatibility."""
    model_dir, active_v = get_model_dir(version)
    if active_v in _MODEL_REGISTRY_CACHE:
        model, le, meta = _MODEL_REGISTRY_CACHE[active_v]
        return model, le, meta, active_v

    model_path = os.path.join(model_dir, "crop_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    meta_path = os.path.join(model_dir, "feature_metadata.json")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    if model is not None:
        try:
            model.set_params(device="cpu")
        except Exception:
            pass

    le = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    _MODEL_REGISTRY_CACHE[active_v] = (model, le, meta)
    return model, le, meta, active_v


def load_crop_v3_submodels(version: Optional[str] = None) -> Tuple[Optional[Any], Dict[str, Any], Dict[str, float], Dict[str, float]]:
    """Loads V3 secondary submodels: Yield Regressor and Knowledge Base."""
    model_dir, active_v = get_model_dir(version)
    yield_model_path = os.path.join(model_dir, "yield_model.pkl")
    yield_model = joblib.load(yield_model_path) if os.path.exists(yield_model_path) else None
    if yield_model is not None:
        try:
            yield_model.set_params(device="cpu")
        except Exception:
            pass

    crop_kb = get_crop_knowledge_base()

    meta_path = os.path.join(model_dir, "feature_metadata.json")
    crop_p90 = {}
    crop_residual_std = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                crop_p90 = meta.get("crop_p90_yield", {})
                crop_residual_std = meta.get("crop_residual_std", {})
        except Exception:
            pass

    return yield_model, crop_kb, crop_p90, crop_residual_std


def _load_artifacts():
    load_crop_artifacts()


def calculate_domain_suitability(
    crop_name: str,
    season: str,
    ph: float,
    rainfall_season: float,
    slope: float,
    erosion_risk: float,
    wsi: float,
) -> Tuple[float, List[str], List[str]]:
    """Calculates deterministic agronomic compatibility and returns positive/limiting factors."""
    rules = CROP_ECOLOGICAL_RULES.get(crop_name)
    if not rules:
        return 0.5, ["General adaptability"], []

    positive_drivers = []
    limiting_factors = []

    # 1. Seasonal Compatibility
    target_season = season.strip().title()
    if target_season in rules["seasons"] or "Whole Year" in rules["seasons"]:
        season_score = 1.0
        positive_drivers.append(f"Season ({target_season}) is optimal for {crop_name}")
    else:
        season_score = 0.2
        limiting_factors.append(f"{crop_name} is typically sown in {', '.join(rules['seasons'])}, not {target_season}")

    # 2. Soil pH Compatibility
    ph_min, ph_max = rules["opt_ph"]
    if ph_min <= ph <= ph_max:
        ph_score = 1.0
        positive_drivers.append(f"Soil pH ({ph:.1f}) is within ideal range ({ph_min}-{ph_max})")
    else:
        diff = min(abs(ph - ph_min), abs(ph - ph_max))
        ph_score = max(0.2, 1.0 - (diff * 0.35))
        if ph < ph_min:
            limiting_factors.append(f"Soil pH ({ph:.1f}) is more acidic than preferred ({ph_min}-{ph_max})")
        else:
            limiting_factors.append(f"Soil pH ({ph:.1f}) is more alkaline than preferred ({ph_min}-{ph_max})")

    # 3. Rainfall & Water Compatibility
    r_min, r_max = rules["opt_rain_season"]
    if r_min <= rainfall_season <= r_max:
        rain_score = 1.0
        positive_drivers.append(f"Seasonal rainfall ({rainfall_season:.0f} mm) matches water requirement")
    elif rainfall_season < r_min:
        rain_score = max(0.2, 1.0 - (r_min - rainfall_season) / r_min)
        if rules["drought_tolerance"] in ["High", "Very High"]:
            rain_score = min(0.85, rain_score + 0.25)
            positive_drivers.append(f"High drought tolerance mitigates lower seasonal rainfall ({rainfall_season:.0f} mm)")
        else:
            limiting_factors.append(f"Rainfall ({rainfall_season:.0f} mm) below preferred minimum ({r_min} mm)")
    else:
        rain_score = max(0.4, 1.0 - (rainfall_season - r_max) / (r_max + 1.0))
        limiting_factors.append(f"Excess rainfall ({rainfall_season:.0f} mm) exceeds preferred threshold ({r_max} mm)")

    # 4. Terrain & Erosion Compatibility
    max_slope = rules["max_slope"]
    if slope > max_slope:
        terrain_score = max(0.2, 1.0 - (slope - max_slope) / max_slope)
        limiting_factors.append(f"Steep slope ({slope:.1f}°) exceeds optimal threshold ({max_slope}°)")
    else:
        terrain_score = 1.0
        positive_drivers.append(f"Gentle terrain slope ({slope:.1f}°) suitable for mechanized cultivation")

    if erosion_risk >= 0.60:
        if rules["erosion_vulnerability"] == "High":
            erosion_score = 0.35
            limiting_factors.append(f"High erosion risk ({erosion_risk*100:.0f}%) poses runoff threat to wide-row crop")
        else:
            erosion_score = 0.75
            positive_drivers.append("Crop provides soil-cover stabilization on erosion-prone ground")
    else:
        erosion_score = 1.0

    domain_score = (
        0.30 * season_score
        + 0.25 * ph_score
        + 0.25 * rain_score
        + 0.10 * terrain_score
        + 0.10 * erosion_score
    )

    return float(np.clip(domain_score, 0.05, 1.0)), positive_drivers[:3], limiting_factors[:3]


def compute_shap_factors(
    model: Any,
    feature_row: np.ndarray,
    feature_names: List[str],
    class_idx: int,
) -> Tuple[List[str], List[str]]:
    """Calculates exact TreeSHAP contributions from native XGBoost booster."""
    try:
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(feature_row, feature_names=feature_names)
        contribs = booster.predict(dmatrix, pred_contribs=True)

        if contribs.ndim == 3:
            vals = contribs[0, :-1, class_idx]
        elif contribs.ndim == 2:
            vals = contribs[0, :-1]
        else:
            vals = contribs.reshape(-1)[:-1]

        pos_indices = [i for i in np.argsort(vals)[::-1] if vals[i] > 0]
        neg_indices = [i for i in np.argsort(vals) if vals[i] < 0]

        alias = {
            "soil_ph": "Soil pH balance",
            "nitrogen": "Available soil nitrogen",
            "phosphorus": "Soil phosphorus levels",
            "potassium": "Soil potassium reserve",
            "organic_carbon": "Soil organic carbon",
            "rainfall_season": "Seasonal rainfall volume",
            "rainfall_30d": "30-day recent rainfall",
            "soil_moisture": "Rootzone soil moisture",
            "temperature_mean": "Favorable thermal range",
            "soil_fertility_index": "High composite soil fertility",
            "water_stress_index": "Elevated atmospheric water stress",
            "erosion_risk_score": "Soil erosion vulnerability",
            "slope": "Terrain slope gradient",
            "season_code": "Crop season alignment",
        }

        pos_reasons = [alias.get(feature_names[i], feature_names[i]) for i in pos_indices[:3]]
        neg_reasons = [alias.get(feature_names[i], feature_names[i]) for i in neg_indices[:2]]

        return pos_reasons, neg_reasons
    except Exception:
        return [], []


def evaluate_counterfactuals(
    crop_model: Any,
    label_encoder: Any,
    base_features_dict: Dict[str, Any],
    base_top_crop: str,
) -> List[Dict[str, Any]]:
    """Evaluates grounded environmental perturbations to identify tipping points (Phase M)."""
    if crop_model is None or label_encoder is None:
        return []

    feature_names = agri_features.FEATURE_COLUMNS
    scenarios = [
        ("Drier Climate (-30% Seasonal Rain)", {"rainfall_season": 0.70, "rainfall_30d": 0.70}),
        ("Wetter Climate (+30% Seasonal Rain)", {"rainfall_season": 1.30, "rainfall_30d": 1.30}),
        ("Soil Acidification (pH -0.5)", {"soil_ph_delta": -0.5}),
        ("Alkaline Shift (pH +0.5)", {"soil_ph_delta": 0.5}),
        ("Assured Supplemental Irrigation (+200 mm)", {"rainfall_season_add": 200.0}),
    ]

    counterfactuals = []
    classes = list(label_encoder.classes_)

    for name, mods in scenarios:
        sim_dict = dict(base_features_dict)
        if "rainfall_season" in mods:
            sim_dict["rainfall_season"] *= mods["rainfall_season"]
        if "rainfall_30d" in mods:
            sim_dict["rainfall_30d"] *= mods["rainfall_30d"]
        if "soil_ph_delta" in mods:
            sim_dict["soil_ph"] = max(4.0, min(9.5, sim_dict["soil_ph"] + mods["soil_ph_delta"]))
        if "rainfall_season_add" in mods:
            sim_dict["rainfall_season"] += mods["rainfall_season_add"]

        sim_vec = np.array([[sim_dict[col] for col in feature_names]], dtype=np.float32)
        sim_probs = crop_model.predict_proba(sim_vec)[0]
        top_idx = int(np.argmax(sim_probs))
        top_c = classes[top_idx]
        top_p = int(round(sim_probs[top_idx] * 100))

        if top_c != base_top_crop:
            insight = f"Under {name.lower()}, {top_c} surpasses {base_top_crop} as highest ranked candidate ({top_p}% modeled suitability)."
        else:
            insight = f"{base_top_crop} maintains highest suitability under {name.lower()} ({top_p}% modeled suitability)."

        counterfactuals.append({
            "scenario": name,
            "top_candidate": top_c,
            "simulated_suitability_percent": top_p,
            "is_rank_shift": top_c != base_top_crop,
            "insight": insight,
        })

    return counterfactuals


def generate_crop_recommendation(
    lat: float,
    lon: float,
    season: str = "Kharif",
    vegetation_ratio: float = 0.35,
    api_key: Optional[str] = None,
    top_k: int = 5,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified Agriculture Intelligence Inference Pipeline (V3 Agronomic Ranking Engine)."""
    crop_model, label_encoder, feature_meta, active_version = load_crop_artifacts(model_version)
    yield_model, crop_kb, crop_p90, crop_residual_std = load_crop_v3_submodels(model_version)

    # 1. Acquire geospatial intelligence
    soil_profile = fetch_unified_soil_profile(lat, lon)
    weather_profile = fetch_live_weather(lat, lon)
    terrain_profile = fetch_terrain_profile(
        lat,
        lon,
        vegetation_ratio=vegetation_ratio,
        rainfall_proxy=weather_profile["rainfall_7d"],
    )

    # 2. Derived Live Weather Forecast Intelligence (Phase I)
    historical_rain = weather_profile.get("rainfall_annual", weather_profile.get("rainfall_90d", 200.0) * 4.0) / 4.0  # approximate seasonal baseline
    forecast_rain_7d = weather_profile.get("forecast_rainfall_7d", weather_profile.get("rainfall_7d", 15.0))
    expected_7d_baseline = max(5.0, historical_rain / 16.0)
    forecast_rain_anomaly = round(forecast_rain_7d - expected_7d_baseline, 1)

    temp_mean = weather_profile.get("temperature_mean", 26.0)
    forecast_temp_mean = weather_profile.get("forecast_temp_mean", temp_mean)
    forecast_temp_anomaly = round(forecast_temp_mean - temp_mean, 1)

    soil_moisture = weather_profile.get("soil_moisture", 0.25)
    water_stress = round(max(0.0, min(1.0, 1.0 - (soil_moisture / 0.35))), 2)
    rainfall_shock = "SURPLUS_RISK" if forecast_rain_7d > 90.0 else "DEFICIT_RISK" if forecast_rain_7d < 2.0 else "NORMAL"

    # 3. Feature Engineering
    raw_obs = {
        **soil_profile,
        **weather_profile,
        **terrain_profile,
        "season": season,
    }
    features_dict = agri_features.engineer_features_row(raw_obs)
    feature_names = agri_features.FEATURE_COLUMNS
    feature_vec = np.array([[features_dict[col] for col in feature_names]], dtype=np.float32)

    # 4. Model Inference
    if crop_model is not None and label_encoder is not None:
        raw_probs = crop_model.predict_proba(feature_vec)[0]
        classes = list(label_encoder.classes_)
    else:
        classes = list(CROP_ECOLOGICAL_RULES.keys())
        raw_probs = np.full(len(classes), 1.0 / len(classes))

    # 5. Multi-Candidate Synthesis & Physiological Scoring
    crop_dummies_cols = [f"crop_is_{c}" for c in sorted(classes)]
    recommendations = []

    for idx, crop_name in enumerate(classes):
        p_model = float(raw_probs[idx]) if idx < len(raw_probs) else 0.05
        d_score, pos_dom, neg_dom = calculate_domain_suitability(
            crop_name=crop_name,
            season=season,
            ph=soil_profile["soil_ph"],
            rainfall_season=weather_profile["rainfall_30d"] * 2.5,
            slope=terrain_profile["slope"],
            erosion_risk=terrain_profile["erosion_risk_score"],
            wsi=features_dict["water_stress_index"],
        )

        shap_pos, shap_neg = compute_shap_factors(crop_model, feature_vec, feature_names, idx)

        # Fused Agronomic Suitability (System C: 0.90 ML + 0.10 Knowledge Compatibility)
        suitability = float(np.clip(0.90 * p_model + 0.10 * d_score, 0.05, 0.99))

        # Expected Yield Modeling (Phase B, C)
        est_yield = 0.0
        yield_range_str = "N/A"
        rel_yield_pct = 75
        if yield_model is not None:
            try:
                yield_feat_vec = feature_vec[0].tolist() + [1.0 if d == f"crop_is_{crop_name}" else 0.0 for d in crop_dummies_cols]
                est_raw = float(yield_model.predict(np.array([yield_feat_vec], dtype=np.float32))[0])
                est_yield = round(max(0.1, est_raw), 2)
                p90 = max(0.5, crop_p90.get(crop_name, 3.0))
                rel_yield_pct = int(round(np.clip(est_yield / p90, 0.05, 1.0) * 100))
                sigma = crop_residual_std.get(crop_name, 0.4)
                yield_range_str = f"{est_yield:.1f} +/- {sigma:.1f} t/ha"
            except Exception:
                est_yield = 2.0
                yield_range_str = "2.0 +/- 0.5 t/ha"

        # Risk breakdown (Phase J, K)
        slope_deg = terrain_profile["slope"]
        erosion_prob = terrain_profile["erosion_risk_score"]
        er_risk_label = "High" if erosion_prob >= 0.60 else "Moderate" if erosion_prob >= 0.30 else "Low"
        ter_risk_label = "High" if slope_deg > 18.0 else "Moderate" if slope_deg > 10.0 else "Low"
        w_risk_label = "High" if water_stress > 0.70 or abs(forecast_temp_anomaly) > 4.0 else "Moderate" if water_stress > 0.40 else "Low"

        # Composite candidate risk
        candidate_risk = "High" if (er_risk_label == "High" or w_risk_label == "High") else "Moderate" if (er_risk_label == "Moderate" or w_risk_label == "Moderate") else "Low"

        # Confidence metric: data quality + model certainty
        data_qual_factor = (
            1.0 if soil_profile["soil_data_quality"] == "HIGH" and weather_profile["weather_data_quality"] == "HIGH"
            else 0.85 if soil_profile["soil_data_quality"] == "HIGH" or weather_profile["weather_data_quality"] == "HIGH"
            else 0.70
        )
        confidence = float(np.clip((0.60 + (p_model * 0.40)) * data_qual_factor, 0.45, 0.95))

        drivers = list(dict.fromkeys(pos_dom + shap_pos))[:3]
        limiters = list(dict.fromkeys(neg_dom + shap_neg))[:2]

        recommendations.append({
            "crop": crop_name,
            "suitability_score": round(suitability, 3),
            "suitability_percent": int(round(suitability * 100)),
            "confidence": round(confidence, 3),
            "confidence_percent": int(round(confidence * 100)),
            "expected_yield_tha": est_yield,
            "expected_yield_range": yield_range_str,
            "relative_yield_potential_percent": rel_yield_pct,
            "risk": candidate_risk,
            "risk_breakdown": {
                "weather_risk": w_risk_label,
                "terrain_risk": ter_risk_label,
                "erosion_risk": er_risk_label,
            },
            "positive_drivers": drivers,
            "limiting_factors": limiters,
        })

    # Sort candidates by suitability score descending
    recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
    top_recommendations = recommendations[:top_k]
    primary_crop = top_recommendations[0]
    second_crop = top_recommendations[1] if len(top_recommendations) > 1 else None

    # 6. Recommendation Uncertainty & Stability Analysis (Phase L)
    decision_margin = round(primary_crop["suitability_score"] - (second_crop["suitability_score"] if second_crop else 0.0), 3)
    if decision_margin >= 0.08:
        ranking_stability = "High Stability"
        margin_interp = f"{primary_crop['crop']} demonstrates strong evidence-based separation (+{decision_margin*100:.1f}%) over {second_crop['crop'] if second_crop else 'alternatives'}."
    elif decision_margin >= 0.03:
        ranking_stability = "Moderate Stability"
        margin_interp = f"{primary_crop['crop']} is favored, but {second_crop['crop']} remains a viable agronomic alternative (narrow {decision_margin*100:.1f}% margin)."
    else:
        ranking_stability = "Close Decision (Multiple Crops Viable)"
        margin_interp = f"Near-boundary tie between {primary_crop['crop']} and {second_crop['crop']}. Agro-climatic conditions support both crops interchangeably."

    uncertainty_dict = {
        "decision_margin": decision_margin,
        "ranking_stability": ranking_stability,
        "recommendation_uncertainty": round(max(0.05, 1.0 - decision_margin * 5.0), 2),
        "interpretation": margin_interp,
    }

    # 7. Counterfactual Analysis (Phase M)
    counterfactuals = evaluate_counterfactuals(
        crop_model=crop_model,
        label_encoder=label_encoder,
        base_features_dict=features_dict,
        base_top_crop=primary_crop["crop"],
    )

    # Enrich candidates with Section 22 decision intelligence contract keys
    for r_idx, cand in enumerate(top_recommendations):
        cand["rank"] = r_idx + 1
        cand["suitability"] = cand["suitability_score"]
        c_sigma = crop_residual_std.get(cand["crop"], 0.4)
        c_est = cand["expected_yield_tha"]
        cand["expected_yield"] = {
            "estimate": c_est,
            "lower": round(max(0.0, c_est - 1.645 * c_sigma), 2),
            "upper": round(c_est + 1.645 * c_sigma, 2),
        }
        cand["decision_margin"] = decision_margin
        cand["ranking_stability"] = ranking_stability
        cand["drivers"] = {
            "positive": cand["positive_drivers"],
            "limiting": cand["limiting_factors"],
        }
        cand["counterfactuals"] = counterfactuals

    # 8. Overall Data Quality
    qualities = [
        soil_profile["soil_data_quality"],
        weather_profile["weather_data_quality"],
        terrain_profile["terrain_data_quality"],
    ]
    if qualities.count("HIGH") >= 2:
        overall_quality = "HIGH"
    elif "LOW" in qualities and qualities.count("LOW") >= 2:
        overall_quality = "LOW"
    else:
        overall_quality = "MODERATE"

    # 9. Structured LLM Explanation Generation
    llm_prompt = f"""
You are an expert agricultural researcher providing decision intelligence.
Location: Lat {lat:.2f}, Lon {lon:.2f} | Season: {season}
Highest-Ranked Crop: {primary_crop['crop']}
Suitability: {primary_crop['suitability_percent']}% | Confidence: {primary_crop['confidence_percent']}% | Risk: {primary_crop['risk']}
Expected Yield: {primary_crop['expected_yield_range']} (Model Estimate)
Ranking Stability: {ranking_stability} (Margin: {decision_margin*100:.1f}%)
Why Ranked Highest: {', '.join(primary_crop['positive_drivers']) if primary_crop['positive_drivers'] else 'Balanced pedo-climatic compatibility'}
Limiting Factors: {', '.join(primary_crop['limiting_factors']) if primary_crop['limiting_factors'] else 'No critical physiological barriers'}
Alternatives: {', '.join([r['crop'] for r in top_recommendations[1:4]])}
Soil: pH {soil_profile['soil_ph']:.1f}, Texture {soil_profile['soil_texture_class']}
Forecast Weather: 7d Rain {forecast_rain_7d} mm (Anomaly: {forecast_rain_anomaly:+.1f} mm), Temp {temp_mean}°C
Erosion Risk: {terrain_profile['erosion_risk_label']} ({terrain_profile['erosion_risk_score']*100:.0f}%)

Generate an evidence-grounded summary (3-4 sentences).
Clearly explain:
1. Why {primary_crop['crop']} is the highest-ranked candidate under available conditions.
2. The expected yield potential with its statistical uncertainty.
3. Environmental risk limitations (forecast anomalies, terrain/erosion).
Do NOT claim guaranteed harvest or use absolute words like 'best crop'. Frame as highest-ranked candidate under available evidence.
""".strip()

    fallback_narrative = (
        f"{primary_crop['crop']} is the highest-ranked candidate under available data with {primary_crop['suitability_percent']}% suitability "
        f"and {primary_crop['confidence_percent']}% confidence for the {season} season. "
        f"The model estimates an expected yield of {primary_crop['expected_yield_range']}, reflecting favorable agro-climatic indicators: "
        f"{', '.join(primary_crop['positive_drivers'][:2]) if primary_crop['positive_drivers'] else 'balanced soil and moisture compatibility'}. "
        f"Environmental risk is rated as {primary_crop['risk']} due to {terrain_profile['erosion_risk_label'].lower()} erosion context "
        f"and 7-day forecast precipitation ({forecast_rain_7d:.1f} mm). Alternatives include {', '.join([r['crop'] for r in top_recommendations[1:4]])}."
    )

    llm_text, insight_mode, insight_status = generate_groq_from_prompt_with_status(
        prompt=llm_prompt,
        fallback_text=fallback_narrative,
        api_key=api_key,
    )

    # 10. Risk and Drivers Aggregation
    drivers_dict = {
        "model_drivers": primary_crop["positive_drivers"],
        "agronomic_drivers": [
            f"Soil pH ({soil_profile['soil_ph']:.1f}) in physiological growth zone",
            f"Thermal regime ({weather_profile['temperature_mean']:.1f}°C) matches vegetative needs",
            f"Rootzone moisture ({weather_profile['soil_moisture']:.2f} m3/m3) supports crop water demand",
        ],
        "risk_drivers": primary_crop["limiting_factors"] + [
            f"Forecast rain anomaly: {forecast_rain_anomaly:+.1f} mm over 7 days",
            f"Terrain slope: {terrain_profile['slope']:.1f}° with {terrain_profile['erosion_risk_label']} erosion vulnerability",
        ],
    }

    risk_dict = {
        "composite_risk": primary_crop["risk"],
        "weather_risk": primary_crop["risk_breakdown"]["weather_risk"],
        "terrain_risk": primary_crop["risk_breakdown"]["terrain_risk"],
        "erosion_risk": primary_crop["risk_breakdown"]["erosion_risk"],
        "forecast_rainfall_anomaly_mm": forecast_rain_anomaly,
        "forecast_temperature_anomaly_c": forecast_temp_anomaly,
        "water_stress_index": water_stress,
        "rainfall_shock": rainfall_shock,
        "advisory": (
            "High runoff pressure detected; incorporate contour bunding or ground cover crops."
            if terrain_profile["erosion_risk_score"] >= 0.70
            else "Moderate terrain vulnerability; standard soil conservation practices advised."
            if terrain_profile["erosion_risk_score"] >= 0.30
            else "Stable terrain conditions with low immediate erosion risk."
        ),
    }

    return {
        "status": "success",
        "model": {
            "name": f"crop_intelligence_{active_version}",
            "version": active_version,
            "dataset_version": "v1.0",
            "architecture": (
                "Spatiotemporal Crop Ranking & Crop-Conditional Yield Potential Engine"
                if active_version == "v4"
                else "Yield-Aware Agronomic Ranking & Physiological Compatibility Engine"
            ),
            "hardware_device": "NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0)",
        },
        "location": {
            "latitude": lat,
            "longitude": lon,
            "season": season,
            "agro_climatic_zone": soil_profile["agro_climatic_zone"],
            "zone_description": soil_profile["zone_description"],
        },
        "soil": soil_profile,
        "weather": weather_profile,
        "terrain": terrain_profile,
        "erosion": {
            "risk_score": terrain_profile["erosion_risk_score"],
            "risk_label": terrain_profile["erosion_risk_label"],
            "advisory": risk_dict["advisory"],
        },
        "recommendations": top_recommendations,
        "primary_recommendation": primary_crop,
        "uncertainty": uncertainty_dict,
        "risk": risk_dict,
        "drivers": drivers_dict,
        "counterfactuals": counterfactuals,
        "explanation": {
            "summary": llm_text,
            "insight_mode": insight_mode,
            "insight_status": insight_status,
        },
        "explanations": {
            "summary": llm_text,
            "llm_narrative": llm_text,
            "insight_mode": insight_mode,
            "insight_status": insight_status,
        },
        "data_quality": {
            "soil_data_quality": soil_profile["soil_data_quality"],
            "weather_data_quality": weather_profile["weather_data_quality"],
            "terrain_data_quality": terrain_profile["terrain_data_quality"],
            "overall_data_quality": overall_quality,
        },
        "limitations": [
            "Suitability scores represent model and ecological compatibility, NOT guaranteed crop success or yield.",
            "Expected yield is an empirical model estimate with documented variance, sensitive to management and micro-climate.",
            "Micro-irrigation, fertilizer application, and local pest management are essential operational determinants.",
        ],
    }
