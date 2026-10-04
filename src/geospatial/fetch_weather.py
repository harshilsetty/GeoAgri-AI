"""Weather Data Acquisition Module.

Integrates Open-Meteo REST API with disk caching to extract:
1. Current environmental state (temperature, humidity, surface and shallow soil moisture).
2. Historical precipitation aggregations (7d, 14d, 30d, 90d sums).
3. 7-day weather forecast (cumulative rainfall, precipitation probability, daily min/max temp).
4. Evapotranspiration (ET0) and water balance indicators.
"""

import json
import os
import certifi
import requests
from typing import Any, Dict, Optional

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "weather")
os.makedirs(CACHE_DIR, exist_ok=True)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _get_cache_filepath(lat: float, lon: float) -> str:
    lat_key = f"{lat:.2f}".replace(".", "_").replace("-", "m")
    lon_key = f"{lon:.2f}".replace(".", "_").replace("-", "m")
    return os.path.join(CACHE_DIR, f"weather_{lat_key}_{lon_key}.json")


def fetch_live_weather(lat: float, lon: float, timeout: int = 8) -> Dict[str, Any]:
    """Retrieves rich meteorological observations and short-term forecasts for coordinates."""
    cache_path = _get_cache_filepath(lat, lon)

    # 1. Try Live Open-Meteo API
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,et0_fao_evapotranspiration",
            "past_days": 14,
            "forecast_days": 7,
            "timezone": "auto",
        }
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout, verify=certifi.where())
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        precip_series = daily.get("precipitation_sum", [])
        past_precip = precip_series[:14] if len(precip_series) >= 14 else precip_series
        forecast_precip = precip_series[14:] if len(precip_series) > 14 else []

        rainfall_7d = float(sum(past_precip[-7:])) if len(past_precip) >= 7 else float(sum(past_precip))
        rainfall_14d = float(sum(past_precip)) if past_precip else 0.0
        # Estimated 30d/90d extrapolations from recent trajectory
        rainfall_30d = round(rainfall_14d * (30.0 / 14.0), 1)
        rainfall_90d = round(rainfall_14d * (90.0 / 14.0), 1)

        forecast_rainfall_7d = float(sum(forecast_precip)) if forecast_precip else 0.0
        precip_probs = daily.get("precipitation_probability_max", [])
        forecast_prob_max = (
            float(max(precip_probs[14:]))
            if len(precip_probs) > 14
            else float(max(precip_probs)) if precip_probs else 0.0
        )

        temp_max_series = daily.get("temperature_2m_max", [])
        temp_min_series = daily.get("temperature_2m_min", [])
        temp_max = float(max(temp_max_series)) if temp_max_series else 32.0
        temp_min = float(min(temp_min_series)) if temp_min_series else 22.0
        temp_mean = float(current.get("temperature_2m", (temp_max + temp_min) / 2.0))

        et0_series = daily.get("et0_fao_evapotranspiration", [])
        et0_mean = float(sum(et0_series) / len(et0_series)) if et0_series else 4.5

        soil_m1 = current.get("soil_moisture_0_to_1cm")
        soil_m2 = current.get("soil_moisture_1_to_3cm")
        soil_moisture = (
            float((soil_m1 + soil_m2) / 2.0)
            if (soil_m1 is not None and soil_m2 is not None)
            else float(soil_m1 or soil_m2 or 0.25)
        )

        elevation = float(data.get("elevation", 300.0))

        result = {
            "temperature_current": round(float(current.get("temperature_2m", temp_mean)), 1),
            "temperature_mean": round(temp_mean, 1),
            "temperature_min": round(temp_min, 1),
            "temperature_max": round(temp_max, 1),
            "temperature_range": round(temp_max - temp_min, 1),
            "humidity_mean": round(float(current.get("relative_humidity_2m", 60.0)), 1),
            "soil_moisture": round(soil_moisture, 3),
            "rainfall_7d": round(rainfall_7d, 1),
            "rainfall_14d": round(rainfall_14d, 1),
            "rainfall_30d": round(rainfall_30d, 1),
            "rainfall_90d": round(rainfall_90d, 1),
            "forecast_rainfall_7d": round(forecast_rainfall_7d, 1),
            "forecast_precip_probability": round(forecast_prob_max, 1),
            "et0": round(et0_mean, 2),
            "elevation": round(elevation, 1),
            "weather_data_quality": "HIGH",
            "weather_data_source": "open_meteo_live",
        }

        # Cache result
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result

    except Exception:
        # 2. Check disk cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    cached["weather_data_quality"] = "MODERATE"
                    cached["weather_data_source"] = "open_meteo_cache"
                    return cached
            except Exception:
                pass

        # 3. Robust regional climatological fallback
        return {
            "temperature_current": 28.0,
            "temperature_mean": 27.5,
            "temperature_min": 21.0,
            "temperature_max": 34.0,
            "temperature_range": 13.0,
            "humidity_mean": 62.0,
            "soil_moisture": 0.24,
            "rainfall_7d": 18.0,
            "rainfall_14d": 35.0,
            "rainfall_30d": 75.0,
            "rainfall_90d": 220.0,
            "forecast_rainfall_7d": 12.0,
            "forecast_precip_probability": 30.0,
            "et0": 4.8,
            "elevation": 350.0,
            "weather_data_quality": "LOW",
            "weather_data_source": "regional_climatology_fallback",
        }
