"""Agricultural Feature Engineering Module (Phase G).

Calculates domain-grounded agronomic derived features:
1. soil_fertility_index: Composite macro-nutrient (N, P, K, OC, pH) fertility score [0, 100].
2. water_stress_index: Ratio of atmospheric evaporative demand (ET0) to rootzone moisture & precipitation.
3. rainfall_deviation: Seasonal precipitation anomaly relative to long-term annual baseline.
4. temperature_range: Daily/seasonal diurnal temperature range (T_max - T_min).
5. soil_moisture_index: Normalized soil moisture relative to field capacity.
6. Season & Soil Texture encodings.
7. Existing erosion model risk score context integration.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "organic_carbon",
    "electrical_conductivity",
    "clay",
    "sand",
    "silt",
    "temperature_mean",
    "temperature_range",
    "humidity_mean",
    "rainfall_season",
    "rainfall_30d",
    "rainfall_90d",
    "soil_moisture",
    "et0",
    "elevation",
    "slope",
    "erosion_risk_score",
    "soil_fertility_index",
    "water_stress_index",
    "rainfall_deviation",
    "soil_moisture_index",
    "season_code",
    "texture_code",
]

SEASON_MAP: Dict[str, int] = {
    "kharif": 0,
    "rabi": 1,
    "summer": 2,
    "whole year": 3,
}

TEXTURE_MAP: Dict[str, int] = {
    "clay": 0,
    "clay loam": 1,
    "loam": 2,
    "sandy loam": 3,
    "sand": 4,
    "silt loam": 5,
    "sandy clay loam": 6,
}


def calculate_soil_fertility_index(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    organic_carbon: float,
    ph: float,
) -> float:
    """Calculates Soil Fertility Index (SFI) on a 0-100 scale using ICAR nutrient ratings."""
    # N score (optimal ~250-350 kg/ha)
    n_score = min(25.0, (nitrogen / 300.0) * 25.0)

    # P score (optimal ~20-35 kg/ha)
    p_score = min(20.0, (phosphorus / 25.0) * 20.0)

    # K score (optimal ~220-320 kg/ha)
    k_score = min(20.0, (potassium / 260.0) * 20.0)

    # OC score (optimal >= 0.75%)
    oc_score = min(20.0, (organic_carbon / 0.75) * 20.0)

    # pH optimality (neutral 6.5-7.5 optimal, deviations penalized)
    ph_penalty = min(15.0, abs(ph - 7.0) * 5.0)
    ph_score = max(0.0, 15.0 - ph_penalty)

    return float(np.clip(n_score + p_score + k_score + oc_score + ph_score, 5.0, 100.0))


def calculate_water_stress_index(
    et0: float,
    rainfall_30d: float,
    soil_moisture: float,
) -> float:
    """Calculates crop water stress index (higher indicates greater water deficit)."""
    monthly_evap = max(10.0, et0 * 30.0)
    moisture_equivalent = max(5.0, rainfall_30d + (soil_moisture * 150.0))
    wsi = monthly_evap / moisture_equivalent
    return float(np.clip(wsi, 0.1, 5.0))


def calculate_rainfall_deviation(
    rainfall_season: float,
    rainfall_annual: float,
) -> float:
    """Calculates normalized rainfall deviation from expected seasonal partition."""
    expected_season_rain = max(50.0, rainfall_annual * 0.65)
    deviation = (rainfall_season - expected_season_rain) / expected_season_rain
    return float(np.clip(deviation, -1.0, 2.0))


def engineer_features_row(data: Dict[str, Any]) -> Dict[str, float]:
    """Engineers all model features from raw observation dict."""
    ph = float(data.get("soil_ph", 7.0))
    n = float(data.get("nitrogen", 180.0))
    p = float(data.get("phosphorus", 15.0))
    k = float(data.get("potassium", 200.0))
    oc = float(data.get("organic_carbon", 0.55))
    ec = float(data.get("electrical_conductivity", 0.3))

    clay = float(data.get("clay", 25.0))
    sand = float(data.get("sand", 45.0))
    silt = float(data.get("silt", 30.0))

    t_mean = float(data.get("temperature_mean", 26.0))
    t_min = float(data.get("temperature_min", 20.0))
    t_max = float(data.get("temperature_max", 32.0))
    t_range = float(data.get("temperature_range", max(1.0, t_max - t_min)))
    humidity = float(data.get("humidity_mean", 60.0))

    rain_annual = float(data.get("rainfall_annual", 950.0))
    rain_season = float(data.get("rainfall_season", rain_annual * 0.65))
    rain_7d = float(data.get("rainfall_7d", rain_season / 16.0))
    rain_30d = float(data.get("rainfall_30d", rain_season / 4.0))
    rain_90d = float(data.get("rainfall_90d", rain_season * 0.8))

    soil_moist = float(data.get("soil_moisture", 0.22))
    et0 = float(data.get("et0", 4.5))

    elevation = float(data.get("elevation", 350.0))
    slope = float(data.get("slope", 5.0))
    erosion_risk = float(data.get("erosion_risk_score", 0.35))

    season_str = str(data.get("season", "Kharif")).strip().lower()
    season_code = SEASON_MAP.get(season_str, 0)

    texture_str = str(data.get("soil_texture_class", "Loam")).strip().lower()
    texture_code = TEXTURE_MAP.get(texture_str, 2)

    # Derived
    sfi = calculate_soil_fertility_index(n, p, k, oc, ph)
    wsi = calculate_water_stress_index(et0, rain_30d, soil_moist)
    rain_dev = calculate_rainfall_deviation(rain_season, rain_annual)
    smi = float(np.clip(soil_moist / 0.40, 0.05, 1.25))

    return {
        "soil_ph": ph,
        "nitrogen": n,
        "phosphorus": p,
        "potassium": k,
        "organic_carbon": oc,
        "electrical_conductivity": ec,
        "clay": clay,
        "sand": sand,
        "silt": silt,
        "temperature_mean": t_mean,
        "temperature_range": t_range,
        "humidity_mean": humidity,
        "rainfall_season": rain_season,
        "rainfall_30d": rain_30d,
        "rainfall_90d": rain_90d,
        "soil_moisture": soil_moist,
        "et0": et0,
        "elevation": elevation,
        "slope": slope,
        "erosion_risk_score": erosion_risk,
        "soil_fertility_index": round(sfi, 2),
        "water_stress_index": round(wsi, 3),
        "rainfall_deviation": round(rain_dev, 3),
        "soil_moisture_index": round(smi, 3),
        "season_code": float(season_code),
        "texture_code": float(texture_code),
    }


def engineer_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineers full tabular feature matrix for model training."""
    rows = []
    for _, raw_row in df.iterrows():
        rows.append(engineer_features_row(raw_row.to_dict()))

    feat_df = pd.DataFrame(rows)[FEATURE_COLUMNS]
    target = df["crop"]
    return feat_df, target
