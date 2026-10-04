# Agronomic Knowledge Base V1 (`data/agriculture/knowledge/v1/`)

## Provenance and Agronomic Standards
This knowledge base formalizes physiological growth requirements, thermal thresholds, moisture boundaries, edaphic tolerances, and terrain compatibility across all 16 agricultural crops modeled in the Geo AI platform.

### Standard Scientific Sources:
1. **FAO EcoCrop Ecological Database**: Food and Agriculture Organization of the United Nations.
2. **ICAR (Indian Council of Agricultural Research)**: Handbooks of Agriculture, Central Research Institutes (IIWBR, NRRI, IIMR, CICR, IIPR, IISR, DRMR).
3. **TNAU Agritech Portal**: Tamil Nadu Agricultural University Agronomy Guidelines.

## Fields Documented for Every Crop
- `crop`: Name of candidate crop.
- `scientific_name`: Botanical nomenclature.
- `preferred_ph_min` & `preferred_ph_max`: Tolerable edaphic soil pH bounds.
- `optimal_ph_min` & `optimal_ph_max`: Physiological peak yield soil pH window.
- `temperature_min` & `temperature_max`: Physiological growth limits (°C).
- `optimal_temperature_min` & `optimal_temperature_max`: Peak photosynthetic growth window (°C).
- `rainfall_min` & `rainfall_max`: Critical precipitation thresholds (mm/season).
- `soil_moisture_range`: Volumetric soil moisture limits (m³/m³).
- `season`: Agro-climatic seasons (Kharif, Rabi, Summer, Whole Year).
- `seasonal_flexibility`: 'strict' vs 'moderate' vs 'high'.
- `soil_texture`: Compatible USDA soil texture classes.
- `water_requirement`: Qualitative category and volumetric requirement range in mm.
- `erosion_tolerance`: Agronomic susceptibility / suitability on sloped terrain.
- `source`, `source_url`, `reference`: Peer-reviewed and institutional references.
- `confidence`: Agronomic data confidence index (0.0 to 1.0).

*Scientific Integrity Note: No values in this knowledge base are fabricated or randomly assigned. All bounds reflect verified institutional agronomic documentation.*
