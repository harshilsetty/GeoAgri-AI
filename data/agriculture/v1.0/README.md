# Indian Agro-Climatic Multi-Season Crop Suitability Dataset (v1.0)

## Overview
This dataset provides grounded multi-variate observations connecting Indian geographic locations, soil fertility, soil physical texture, seasonal weather distributions, terrain elevation/slope, and soil erosion risk with cultivated crop outcomes.

- **Total Records:** 10091
- **Number of Features:** 31
- **Target Variable:** `crop` (16 distinct agricultural crops)
- **Temporal Span:** 1997 - 2020

## Supported Crops
Chickpea, Cotton, Finger Millet, Groundnut, Maize, Moong, Mustard, Pearl Millet, Pigeonpea, Rice, Sesamum, Sorghum, Soybean, Sugarcane, Urad, Wheat

## Feature Schema
| Feature | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `latitude` | float | degrees N | Geographic latitude centroid |
| `longitude` | float | degrees E | Geographic longitude centroid |
| `state` | string | text | Indian State name |
| `season` | string | category | Crop season (Kharif, Rabi, Summer, Whole Year) |
| `soil_ph` | float | pH units | Soil pH in water (depth 0-15cm) |
| `nitrogen` | float | kg/ha | Available Soil Nitrogen (N) |
| `phosphorus` | float | kg/ha | Available Soil Phosphorus (P) |
| `potassium` | float | kg/ha | Available Soil Potassium (K) |
| `organic_carbon` | float | % | Soil Organic Carbon |
| `electrical_conductivity` | float | dS/m | Soil Electrical Conductivity |
| `clay` | float | % | Clay particle percentage (0-5cm) |
| `sand` | float | % | Sand particle percentage (0-5cm) |
| `silt` | float | % | Silt particle percentage (0-5cm) |
| `soil_texture_class` | string | category | USDA soil texture category |
| `temperature_mean` | float | °C | Seasonal mean air temperature |
| `temperature_min` | float | °C | Seasonal minimum air temperature |
| `temperature_max` | float | °C | Seasonal maximum air temperature |
| `humidity_mean` | float | % | Relative humidity |
| `rainfall_annual` | float | mm | Annual precipitation |
| `rainfall_season` | float | mm | Seasonal precipitation |
| `rainfall_7d` | float | mm | 7-day cumulative rainfall |
| `rainfall_30d` | float | mm | 30-day cumulative rainfall |
| `rainfall_90d` | float | mm | 90-day cumulative rainfall |
| `soil_moisture` | float | m³/m³ | Rootzone soil moisture |
| `et0` | float | mm/day | FAO Penman-Monteith reference ET0 |
| `elevation` | float | meters | Ground elevation above sea level |
| `slope` | float | degrees | Terrain slope angle |
| `erosion_risk_score` | float | [0, 1] | XGBoost model erosion risk score |
| `crop` | string | category | Cultivated crop label (Target) |
| `yield` | float | tons/ha | Historical crop yield |

## Provenance
- Directorate of Economics and Statistics (DES), Ministry of Agriculture & Farmers Welfare, India
- ICRISAT District-Level Database (DLD)
- Indian Council of Agricultural Research (ICAR) & Soil Health Card Scheme
- Open-Meteo API
- Open-Elevation API & Geo AI Erosion Pipeline
