import certifi
import numpy as np
import requests


def _fallback(value, default):
    return float(value) if value is not None else float(default)


def get_real_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=5, verify=certifi.where())
        response.raise_for_status()
        return float(response.json()["results"][0]["elevation"])
    except Exception:
        return _fallback(None, 400.0)


def get_avg_rainfall(lat, lon):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=precipitation_sum&past_days=7"
        )
        response = requests.get(url, timeout=5, verify=certifi.where())
        response.raise_for_status()
        rainfall = response.json()["daily"]["precipitation_sum"]
        return float(sum(rainfall) / len(rainfall)) if rainfall else 0.0
    except Exception:
        return _fallback(None, 25.0)


def get_soil_type(lat, lon):
    try:
        url = (
            "https://rest.isric.org/soilgrids/v2.0/properties/query?"
            f"lat={lat}&lon={lon}&property=clay&depth=0-5cm"
        )
        response = requests.get(url, timeout=5, verify=certifi.where())
        response.raise_for_status()
        clay = response.json()["properties"]["layers"][0]["depths"][0]["values"]["mean"]
        return 3 if clay > 40 else 2 if clay > 20 else 1
    except Exception:
        return 2


def get_slope(lat, lon):
    try:
        base = get_real_elevation(lat, lon)
        north = get_real_elevation(lat + 0.001, lon)
        east = get_real_elevation(lat, lon + 0.001)
        if base == 400.0 and north == 400.0 and east == 400.0:
            return 15.0
        return float((abs(north - base) + abs(east - base)) / 2)
    except Exception:
        return _fallback(None, 15.0)


def extract_full_features(
    lat,
    lon,
    vegetation_ratio=0.0,
    boulders_ratio=0.0,
    ruins_ratio=0.0,
    structures_ratio=0.0,
):
    del boulders_ratio, ruins_ratio, structures_ratio
    return {
        "slope": get_slope(lat, lon),
        "rainfall": get_avg_rainfall(lat, lon),
        "soil": get_soil_type(lat, lon),
        "elevation": get_real_elevation(lat, lon),
        "vegetation": float(vegetation_ratio),
    }