# ?? GeoAgri-AI Dataset Architecture & Provenance

This directory contains versioned datasets, reference coordinates, and agronomic knowledge bases supporting GeoAgri-AI's geospatial agricultural intelligence pipeline.

---

## 1. Directory Structure

```text
data/
??? README.md                          # This file
??? agriculture/
    ??? knowledge/
    ?   ??? v1/
    ?       ??? crop_knowledge_base.json   # FAO EcoCrop & ICAR physiological rule base
    ?       ??? README.md                  # Knowledge base documentation & agronomic bounds
    ??? v1.0/
        ??? README.md                      # Dataset version release notes
        ??? metadata.json                  # Schema and feature definitions
        ??? dataset_validation_report.json # Integrity, distribution, and completeness metrics
        ??? processed/
        ?   ??? crop_suitability_dataset.csv # Curated district-level multi-feature dataset
        ??? raw/
            ??? crop_yield_data.csv        # Historical yield observations across Indian districts
            ??? lat_lon_india_district.csv # Spatial centroid coordinate mapping for Indian districts
```

---

## 2. Dataset Purpose & Descriptions

### A. Agronomic Knowledge Base (`knowledge/v1/crop_knowledge_base.json`)
* **Purpose**: Serves as the ground-truth physiological rule engine for the FAO EcoCrop and ICAR compatibility filters.
* **Content**: 22 crops mapped with cardinal growth thresholds:
  * Minimum, optimal, and maximum soil pH
  * Temperature ranges (cardinal baseline, optimal, and upper lethal limits)
  * Rainfall / water requirement envelopes (mm per season)
  * Photoperiod & season alignment (Kharif, Rabi, Zaid, Whole Year)
  * Soil texture tolerance (Sand, Sandy Loam, Loam, Clay Loam, Clay)
* **Source**: Synthesized from Food and Agriculture Organization (FAO) EcoCrop database and Indian Council of Agricultural Research (ICAR) crop advisories.

### B. District Geolocation Index (`v1.0/raw/lat_lon_india_district.csv`)
* **Purpose**: Maps administrative district names to representative latitude and longitude centroids across India.
* **Features**: `State`, `District`, `Latitude`, `Longitude`.

### C. Historical Crop Suitability & Yield Datasets (`v1.0/`)
* **Processed Dataset (`crop_suitability_dataset.csv`)**: 1.88 MB curated tabular dataset containing 26 agronomic features (soil macronutrients, meteorological indices, terrain slope, elevation, and suitability labels).
* **Raw Yield Data (`crop_yield_data.csv`)**: 1.6 MB historical yield observations (t/ha) across Indian states and districts from the Ministry of Agriculture & Farmers Welfare (DAC&FW).

---

## 3. Preprocessing Pipeline

Raw agricultural records undergo the following validation and transformation pipeline via `scripts/data/`:
1. **Coordinate & Spatial Alignment**: Coordinates are verified against Indian bounding boxes (`8.0?N - 37.0?N`, `68.0?E - 97.5?E`).
2. **Feature Imputation & Zonal Benchmarking**: Where physical soil measurements are sparse, depth-stratified ISRIC SoilGrids v2.0 queries are combined with ICAR Soil Health Card zonal baselines.
3. **Agronomic Metric Derivation**:
   * Soil Fertility Index ($SFI \in [0, 100]$)
   * Water Stress Index ($WSI$)
   * Rainfall Deviation ($RD$)
   * Soil Moisture Index ($SMI$)
4. **Validation**: Certified by `scripts/data/validate_dataset.py` ensuring zero target leakage, bounded variances, and clean class distribution.

---

## 4. Live API Ingestion & Cache Policy

In addition to static datasets, GeoAgri-AI connects to external real-time providers:
* **Open-Meteo REST API**: Weather observations, short-term precipitation forecasts, and FAO-56 reference evapotranspiration ($ET_0$).
* **ISRIC SoilGrids v2.0**: Global 250m gridded soil physical properties.
* **Open-Elevation DEM**: Digital Elevation Model for elevation and terrain slope calculations.

*Cache Policy*: Dynamic queries are cached locally in `data/cache/` to ensure resilience against rate limits and network degradation. Local cache files are explicitly git-ignored to prevent storing transient payloads.

---

## 5. Licensing & Data Usage

* FAO EcoCrop information is provided for public academic and non-commercial research use under FAO terms.
* Indian agricultural statistics and Soil Health Card baselines are governed by Government of India Open Government Data (OGD) Platform India (data.gov.in).
* All curated tables and schema definitions in this repository are released under the project's MIT License.
