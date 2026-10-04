# ?? GeoAgri-AI: Solution Overview & System Concept

## 1. Conceptual Framework

GeoAgri-AI is an **intelligent geospatial agricultural decision-support system** engineered to bridge the gap between complex multi-modal spatial data and actionable farm-level recommendations.

The project operates under the fundamental operational doctrine:

```text
Observe  ????  Understand  ????  Reason  ????  Decide  ????  Recommend  ????  Adapt
(Data Ingestion) (Feature Engine)  (Knowledge Engine) (Multi-Objective)  (Advisory)  (Feedback Loop)
```

---

## 2. Multi-Modal Intelligence Layers

### A. Geospatial & Environmental Observation
GeoAgri-AI ingests multi-source geospatial context based on geographic coordinates:
* **Soil Physical & Chemical Properties**: 0-5cm depth-stratified physical parameters (sand, silt, clay, CEC) from ISRIC SoilGrids v2.0 coupled with Indian Council of Agricultural Research (ICAR) Soil Health Card zonal fertility baselines (N, P, K, Organic Carbon, Electrical Conductivity, pH).
* **Atmospheric State & Short-Term Forecasts**: Hourly and daily temperature, relative humidity, surface moisture, 7d/14d/30d/90d historical precipitation aggregates, 7-day rainfall forecasts, and FAO-56 reference evapotranspiration ($ET_0$) via Open-Meteo.
* **Terrain Geomorphology**: Digital Elevation Model (DEM) data yielding elevation, slope gradient, and contextual soil erosion vulnerability.

### B. Computer Vision & Remote Sensing Layer `[In Development / Planned]`
* **Satellite Multispectral Ingestion**: Ingesting Sentinel-2 and Landsat imagery to derive vegetative indices (NDVI, NDRE, EVI, NDWI).
* **Canopy Segmentation & Stress Detection**: Convolutional and Transformer backbones trained to delineate crop field boundaries, evaluate fractional vegetation cover, and isolate localized moisture stress anomalies.

### C. Agronomic Knowledge-Grounded Reasoning
Unlike opaque black-box classifiers, GeoAgri-AI anchors its predictions in verified agronomic science:
* **FAO EcoCrop & ICAR Physiological Rule Engine**: Hard and soft physiological boundaries (temperature limits, pH tolerance, water envelope, photoperiod compatibility) filter candidate recommendations.
* **Multi-Objective Decision Intelligence**: Recommends crops by balancing machine learning classification probabilities ($w=0.45$), soil fertility compatibility ($w=0.20$), seasonal timing ($w=0.15$), climate envelope safety ($w=0.10$), and terrain erosion safety ($w=0.10$).

### D. Explainable & Actionable Delivery
* **TreeSHAP Attributions**: For every ranked crop, exact Shapley values highlight the primary positive drivers (e.g., optimal potassium level, favorable rainfall) and limiting factors (e.g., high water stress index, alkaline pH).
* **LLM-Powered Agronomic Explanations**: Generates farmer-friendly narratives using Groq LLaMA-3.1-8B-Instant with strict ground-truth prompt injection and guaranteed local rule-based fallback.
* **Modern Web Interface**: Interactive Next.js map picker, crop suitability radar charts, SHAP breakdown bars, and downloadable PDF agronomic audit reports.
