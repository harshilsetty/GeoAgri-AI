"""Soil Data Acquisition Module.

Integrates:
1. ISRIC SoilGrids v2.0 REST API with disk caching for depth-stratified physical properties
   (clay, sand, silt, phh2o, soc, nitrogen, cec at depth 0-5cm).
2. Indian Soil Health Card (DAC&FW / ICAR) Agro-Climatic Zone Benchmark Soil Profiles
   for chemical fertility parameters (N, P, K, Organic Carbon, Electrical Conductivity).
"""

import json
import os
import certifi
import requests
from typing import Any, Dict, Optional, Tuple

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "soil")
os.makedirs(CACHE_DIR, exist_ok=True)

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# ICAR / DAC&FW Indian Soil Health Card Agro-Climatic Zone reference fertility baselines
# Derived from published Soil Health Card Portal and ICAR-IISS (Indian Institute of Soil Science)
# zonal soil fertility status surveys across Indian agro-climatic regions.
AGRO_CLIMATIC_SOIL_PROFILES: Dict[str, Dict[str, Any]] = {
    "gangetic_alluvial": {
        "description": "Indo-Gangetic Alluvial Plain (Punjab, Haryana, UP, Bihar, WB)",
        "ph": 7.4,
        "n_kg_ha": 210.0,
        "p_kg_ha": 18.5,
        "k_kg_ha": 240.0,
        "oc_pct": 0.52,
        "ec_ds_m": 0.35,
        "clay_pct": 24.0,
        "sand_pct": 42.0,
        "silt_pct": 34.0,
        "texture": "Clay Loam",
    },
    "black_cotton_vertisol": {
        "description": "Deccan & Central Black Soil Plateau (Maharashtra, MP, Gujarat, Karnataka)",
        "ph": 7.8,
        "n_kg_ha": 175.0,
        "p_kg_ha": 14.0,
        "k_kg_ha": 310.0,
        "oc_pct": 0.65,
        "ec_ds_m": 0.28,
        "clay_pct": 48.0,
        "sand_pct": 22.0,
        "silt_pct": 30.0,
        "texture": "Clay",
    },
    "red_sandy_alfisol": {
        "description": "Southern & Eastern Red Soils (Telangana, AP, Tamil Nadu, Karnataka, Odisha)",
        "ph": 6.5,
        "n_kg_ha": 180.0,
        "p_kg_ha": 16.0,
        "k_kg_ha": 195.0,
        "oc_pct": 0.45,
        "ec_ds_m": 0.18,
        "clay_pct": 18.0,
        "sand_pct": 62.0,
        "silt_pct": 20.0,
        "texture": "Sandy Loam",
    },
    "arid_desert": {
        "description": "Western Arid Region (Rajasthan, North Gujarat)",
        "ph": 8.2,
        "n_kg_ha": 130.0,
        "p_kg_ha": 11.0,
        "k_kg_ha": 280.0,
        "oc_pct": 0.22,
        "ec_ds_m": 0.55,
        "clay_pct": 10.0,
        "sand_pct": 78.0,
        "silt_pct": 12.0,
        "texture": "Sand",
    },
    "laterite_coastal": {
        "description": "West Coast & Ghats Laterite Soils (Kerala, Konkan, Coastal Karnataka)",
        "ph": 5.4,
        "n_kg_ha": 240.0,
        "p_kg_ha": 12.0,
        "k_kg_ha": 160.0,
        "oc_pct": 1.15,
        "ec_ds_m": 0.12,
        "clay_pct": 32.0,
        "sand_pct": 46.0,
        "silt_pct": 22.0,
        "texture": "Sandy Clay Loam",
    },
    "himalayan_hill": {
        "description": "Himalayan Forest and Mountain Soils (Uttarakhand, HP, J&K, Assam hills)",
        "ph": 5.8,
        "n_kg_ha": 290.0,
        "p_kg_ha": 22.0,
        "k_kg_ha": 210.0,
        "oc_pct": 1.45,
        "ec_ds_m": 0.10,
        "clay_pct": 20.0,
        "sand_pct": 52.0,
        "silt_pct": 28.0,
        "texture": "Loam",
    },
}


def _resolve_agro_climatic_zone(lat: float, lon: float) -> str:
    """Classifies geographic coordinates into primary Indian Agro-Climatic Zone."""
    if lat > 28.0 and lon < 80.0:
        if lat > 30.5 or (lat > 29.5 and lon > 78.0):
            return "himalayan_hill"
        if lon < 74.0:
            return "arid_desert"
        return "gangetic_alluvial"
    elif lat >= 23.5:
        if lon < 74.0:
            return "arid_desert"
        elif lon > 85.0 and lat > 25.0:
            return "gangetic_alluvial"
        elif lon >= 74.0 and lon <= 82.0:
            return "black_cotton_vertisol"
        else:
            return "gangetic_alluvial"
    elif lat >= 15.0:
        if lon < 74.5:
            return "laterite_coastal"
        elif lon <= 78.5:
            return "black_cotton_vertisol"
        else:
            return "red_sandy_alfisol"
    else:
        if lon < 76.5:
            return "laterite_coastal"
        return "red_sandy_alfisol"


def _get_cache_filepath(lat: float, lon: float) -> str:
    lat_key = f"{lat:.2f}".replace(".", "_").replace("-", "m")
    lon_key = f"{lon:.2f}".replace(".", "_").replace("-", "m")
    return os.path.join(CACHE_DIR, f"soil_{lat_key}_{lon_key}.json")


def fetch_soilgrids_properties(lat: float, lon: float, timeout: int = 6) -> Optional[Dict[str, float]]:
    """Queries SoilGrids v2.0 REST API for 0-5cm layer properties.
    Returns normalized metrics or None on timeout/error.
    """
    cache_path = _get_cache_filepath(lat, lon)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("data")
        except Exception:
            pass

    params = [
        ("lat", str(lat)),
        ("lon", str(lon)),
        ("property", "phh2o"),
        ("property", "clay"),
        ("property", "sand"),
        ("property", "silt"),
        ("property", "soc"),
        ("property", "nitrogen"),
        ("property", "cec"),
        ("depth", "0-5cm"),
    ]

    try:
        response = requests.get(SOILGRIDS_URL, params=params, timeout=timeout, verify=certifi.where())
        response.raise_for_status()
        payload = response.json()
        layers = payload.get("properties", {}).get("layers", [])

        extracted = {}
        for layer in layers:
            name = layer.get("name")
            depths = layer.get("depths", [])
            if depths and "values" in depths[0] and "mean" in depths[0]["values"]:
                raw_mean = depths[0]["values"]["mean"]
                if raw_mean is not None:
                    extracted[name] = float(raw_mean)

        if not extracted:
            return None

        # SoilGrids units conversion:
        # phh2o: pH * 10 -> divide by 10
        # clay, sand, silt: g/kg (permille) -> divide by 10 for %
        # soc: dg/kg -> divide by 100 for %
        # nitrogen: cg/kg -> convert to mg/kg
        # cec: mmol(c)/kg -> cmol(c)/kg (divide by 10)
        result = {
            "soil_ph": round(extracted["phh2o"] / 10.0, 2) if "phh2o" in extracted else None,
            "clay": round(extracted["clay"] / 10.0, 1) if "clay" in extracted else None,
            "sand": round(extracted["sand"] / 10.0, 1) if "sand" in extracted else None,
            "silt": round(extracted["silt"] / 10.0, 1) if "silt" in extracted else None,
            "organic_carbon": round(extracted["soc"] / 100.0, 2) if "soc" in extracted else None,
            "nitrogen_g_kg": round(extracted["nitrogen"] / 100.0, 2) if "nitrogen" in extracted else None,
            "cec": round(extracted["cec"] / 10.0, 1) if "cec" in extracted else None,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"data": result, "source": "soilgrids_api"}, f, indent=2)

        return result
    except Exception:
        return None


def fetch_unified_soil_profile(
    lat: float,
    lon: float,
    state: Optional[str] = None,
    district: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves full soil fertility + texture profile combining SoilGrids and
    validated Indian Soil Health Card agro-climatic zone standards.
    """
    zone_key = _resolve_agro_climatic_zone(lat, lon)
    zone_defaults = AGRO_CLIMATIC_SOIL_PROFILES.get(zone_key, AGRO_CLIMATIC_SOIL_PROFILES["gangetic_alluvial"])

    soilgrids_data = fetch_soilgrids_properties(lat, lon)
    source = "soilgrids_v2" if soilgrids_data else "soil_health_card_benchmark"
    data_quality = "HIGH" if soilgrids_data else "MODERATE"

    # Merge physical properties (SoilGrids prioritized, Zone fallback)
    ph = soilgrids_data.get("soil_ph") if soilgrids_data and soilgrids_data.get("soil_ph") else zone_defaults["ph"]
    clay = soilgrids_data.get("clay") if soilgrids_data and soilgrids_data.get("clay") else zone_defaults["clay_pct"]
    sand = soilgrids_data.get("sand") if soilgrids_data and soilgrids_data.get("sand") else zone_defaults["sand_pct"]
    silt = soilgrids_data.get("silt") if soilgrids_data and soilgrids_data.get("silt") else zone_defaults["silt_pct"]
    oc = (
        soilgrids_data.get("organic_carbon")
        if soilgrids_data and soilgrids_data.get("organic_carbon")
        else zone_defaults["oc_pct"]
    )

    # Chemical fertility (Soil Health Card provides N, P, K in kg/ha, EC in dS/m)
    nitrogen = zone_defaults["n_kg_ha"]
    phosphorus = zone_defaults["p_kg_ha"]
    potassium = zone_defaults["k_kg_ha"]
    ec = zone_defaults["ec_ds_m"]

    # Soil texture classification
    texture = zone_defaults["texture"]
    if clay >= 40:
        texture = "Clay"
    elif sand >= 70:
        texture = "Sandy"
    elif clay >= 25 and sand <= 45:
        texture = "Clay Loam"
    elif sand >= 45 and clay <= 20:
        texture = "Sandy Loam"
    elif silt >= 50:
        texture = "Silt Loam"
    else:
        texture = "Loam"

    return {
        "soil_ph": float(ph),
        "nitrogen": float(nitrogen),
        "phosphorus": float(phosphorus),
        "potassium": float(potassium),
        "organic_carbon": float(oc),
        "clay": float(clay),
        "sand": float(sand),
        "silt": float(silt),
        "electrical_conductivity": float(ec),
        "soil_texture_class": texture,
        "agro_climatic_zone": zone_key,
        "zone_description": zone_defaults["description"],
        "soil_data_source": source,
        "soil_data_quality": data_quality,
        "depth_layer": "0-5cm (physical), 0-15cm (rootzone fertility)",
    }
