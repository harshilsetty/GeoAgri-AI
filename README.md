# ?? GeoAgri AI

> **Intelligent Geospatial Agricultural Decision-Support System: AI for Crop Monitoring, Moisture Stress Detection & Irrigation Advisory**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Target: KPIT Sparkle 2027](https://img.shields.io/badge/KPIT_Sparkle-2027-red.svg)](https://www.kpit.com/sparkle/)

---

## Overview

**GeoAgri AI** is an intelligent geospatial agricultural decision-support system designed to transform spatial, environmental, and crop-related observations into actionable agricultural insights. Developed as an advanced engineering foundation for the **KPIT Sparkle 2027** competition, GeoAgri AI bridges the gap between raw multi-modal geospatial data and real-world farm management.

The project targets the long-term operational closed loop:

```text
Observe  ????  Understand  ????  Reason  ????  Decide  ????  Recommend  ????  Learn
```

---

## Problem

Modern agricultural operations worldwide face compounded climate and resource crises:
1. **Suboptimal Crop Selection**: Farmers frequently select crops based on tradition or immediate market spikes rather than micro-climate suitability, leading to lower yields, nutrient exhaustion, and economic vulnerability.
2. **Invisible Moisture Stress**: Water deficit damages crop physiology (stomatal closure, cell shrinkage, stunted biomass) days before visible foliar wilting occurs, by which time significant yield loss is irreversible.
3. **Inefficient Irrigation Practices**: Flood and heuristic timer-based irrigation waste over 40% of agricultural freshwater, deplete groundwater aquifers, and cause nutrient leaching, while under-irrigation induces severe harvest penalties.

---

## Proposed Solution

GeoAgri AI addresses these challenges through a unified multi-modal intelligence platform:
* **Multi-Depth Ingestion**: Automatically synthesizes 0-5cm physical soil properties (ISRIC SoilGrids), ICAR zonal chemical fertility baselines, live weather observations, historical rainfall aggregations, and 7-day precipitation forecasts.
* **Domain Feature Engineering**: Synthesizes 26 domain-grounded agronomic indicators including Soil Fertility Index ($SFI$), Water Stress Index ($WSI$), Rainfall Deviation ($RD$), and Soil Moisture Index ($SMI$).
* **Agronomic Knowledge-Grounded Inference**: Pairs calibrated machine learning models (XGBoost) with FAO EcoCrop physiological rules and ICAR agro-climatic boundaries to deliver multi-objective suitability rankings, yield productivity estimations, and model-supported counterfactual explanations.

---

## Why GeoAgri AI?

| Conventional Approach | GeoAgri AI Approach |
| :--- | :--- |
| Single-parameter rules (e.g., rainfall only) | Multi-modal fusion (soil physics + chemistry + weather + terrain + knowledge) |
| Opaque black-box ML predictions | Exact TreeSHAP feature attributions and counterfactual reasoning |
| Reactive irrigation after visual wilting | Predictive moisture stress indicators and evapotranspiration modeling |
| Heuristic crop selection | Multi-objective Pareto optimization balancing suitability, yield, and risk |
| Static laboratory soil testing | Dynamic geospatial integration with calibrated zonal baselines |

---

## Core Capabilities

1. **Precision Geospatial Ingestion**: Point-based coordinate lookup retrieving depth-stratified soil, live weather, short-term forecasts, elevation, and terrain slope.
2. **Multi-Objective Crop Recommendation**: Evaluates candidate crops across model probabilities, soil compatibility, seasonal alignment, climate envelope safety, and terrain erosion risk.
3. **Yield Productivity Estimation**: Secondary regression models predict expected crop yield (t/ha) and relative yield potential compared to regional agro-climatic benchmarks.
4. **Explainable AI & Actionable Insights**: TreeSHAP breakdown of positive drivers and limiting factors, accompanied by Groq LLaMA-3.1 grounded agronomic narratives and deterministic rule-based fallbacks.
5. **Interactive Farm Intelligence Dashboard**: Modern Next.js interface with map location picker, radar charts, suitability rankings, and audit-ready report generation.

---

## System Architecture

```text
               User / Farm Coordinate Request
              (Latitude, Longitude, Season)
                            ?
       ???????????????????????????????????????????
       ?                                         ?
External Geo Services                   Local Fallback / Caches
??? Open-Meteo REST API                 ??? Soil Cache (data/cache/soil/)
??? ISRIC SoilGrids v2.0                ??? ICAR Soil Health Card
??? Open-Elevation DEM                      Agro-Climatic Benchmarks
       ?                                         ?
       ???????????????????????????????????????????
                            ?
               Geospatial Feature Ingestion
       ??? Soil: N, P, K, pH, OC, EC, Sand, Silt, Clay
       ??? Weather: T_mean, T_min, T_max, Humidity, ET0, Rain_7d/30d/90d
       ??? Forecast: Rain_7d, Precip_Prob, Soil Moisture
       ??? Terrain: Elevation, Slope Gradient, Advisory Erosion Context
                            ?
                            ?
               Agricultural Feature Engineering (26D)
       (src/preprocessing/agri_features.py)
       ??? Soil Fertility Index (SFI) ? Water Stress Index (WSI)
       ??? Rainfall Deviation (RD)    ? Soil Moisture Index (SMI)
       ??? Diurnal Temperature Range  ? Terrain Risk Score Context
                            ?
       ???????????????????????????????????????????
       ?                                         ?
Production Crop Model (models/crop/)    Agronomic Knowledge Base
Multi-Class XGBoost Classifier (16-22)  FAO EcoCrop & ICAR Physiological Rules
       ?                                         ?
       ???????????????????????????????????????????
                            ?
          Multi-Objective Compatibility Engine
       Weighted Composite: Model Prob (0.45) + Soil (0.20) + Season (0.15)
                           + Climate Envelope (0.10) + Erosion Safety (0.10)
                            ?
                            ?
          Explainability & Reasoning Layer
       ??? Exact TreeSHAP Driver & Limiting Factor Attribution
       ??? Model-Supported Counterfactual Analysis
       ??? Groq LLaMA-3.1-8B Agronomic Narrative (Local Rule Fallback)
                            ?
                            ?
          Client Delivery & Application Layer
       ??? FastAPI Backend (src/api/backend_api.py)
       ??? Next.js 14 Interactive Web Dashboard (ui/)
```

---

## AI/ML Approach

* **Supervised Gradient Boosted Decision Trees (XGBoost)**: Multi-class classification over 26 domain features for crop suitability across 16-22 crop categories.
* **Secondary Yield Regressors**: XGBoost regressors trained on historical Indian district yield records to estimate harvest productivity (t/ha).
* **Learning-to-Rank (LTR)**: Pairwise and listwise ranking formulations optimizing NDCG and MRR for prioritized crop recommendations.
* **TreeSHAP Explainability**: Local attribution identifying exact feature contributions per inference.
* **LLM Reasoning**: Zero-hallucination prompt synthesis using Groq LLaMA-3.1-8B-Instant strictly conditioned on TreeSHAP attributions and FAO EcoCrop boundaries.

---

## Current Implementation vs. In Development vs. Planned

To maintain scientific integrity and transparency, project status is strictly categorized:

### ?? CURRENTLY IMPLEMENTED (Verified in Repository)
- [x] Geospatial data acquisition pipelines for ISRIC SoilGrids, Open-Meteo, Open-Elevation, and ICAR zonal benchmarks.
- [x] 26-dimensional domain agronomic feature engineering module (`src/preprocessing/agri_features.py`).
- [x] Versioned crop model registry (`models/crop/v1` through `v5`) with pre-trained XGBoost classifiers, yield models, and ranking models.
- [x] FAO EcoCrop and ICAR physiological rule engine (`data/agriculture/knowledge/v1/crop_knowledge_base.json`).
- [x] Multi-objective decision weighting and ranking stability uncertainty computation.
- [x] Exact TreeSHAP attribution and counterfactual analysis engine (`src/models/agri_inference.py`).
- [x] Hybrid Groq LLaMA-3.1 LLM explanation engine with deterministic local fallback (`src/explainability/ai_explainer.py`).
- [x] Production FastAPI backend with `/health`, `/debug/model`, and `/api/agriculture/recommend` endpoints (`src/api/backend_api.py`).
- [x] Next.js 14 web dashboard with interactive map picker, risk radar, and crop intelligence panels (`ui/`).
- [x] Comprehensive 79-test automated test suite with 100% pass rate.

### ?? IN DEVELOPMENT (Active Milestone Work)
- [ ] Automated Sentinel-2 multispectral tile download and cloud-masking scripts.
- [ ] Field-level vegetation index computation pipeline (NDVI, NDRE, EVI, NDWI).
- [ ] Dynamic rootzone water balance module integrating FAO-56 dual crop coefficient formulas.

### ?? PLANNED CAPABILITIES (KPIT Sparkle 2027 Roadmap)
- [ ] Computer vision canopy segmentation for field boundary extraction using DeepLabV3+.
- [ ] Automated early-onset moisture stress detection from thermal and shortwave infrared satellite bands.
- [ ] Dynamic, weather-forecast-driven irrigation schedule advisory (timing and volumetric water recommendations).
- [ ] IoT soil sensor telemetry ingestion for real-time field-level ground truth calibration.
- [ ] Adaptive reinforcement learning feedback loop incorporating seasonal farmer yield outcomes.

---

## Technology Stack

* **Core Logic & Machine Learning**: Python 3.10+, NumPy, Pandas, Scikit-Learn, XGBoost, SHAP, Joblib
* **Geospatial & APIs**: ISRIC SoilGrids REST API, Open-Meteo API, Open-Elevation DEM, Requests, Certifi
* **Backend API**: FastAPI, Uvicorn, Pydantic
* **Frontend Web Application**: Next.js 14, React 18, TypeScript, TailwindCSS, Leaflet, Framer Motion
* **Explainability & Generative AI**: TreeSHAP, Groq API (LLaMA-3.1-8B-Instant)
* **Testing & Quality Assurance**: Pytest, Pytest-Asyncio, HTTPX

---

## Project Structure

```text
GeoAgri-AI/
??? README.md                      # Comprehensive project documentation
??? LICENSE                        # MIT License
??? .gitignore                     # Git hygiene exclusions
??? requirements.txt               # Certified Python dependencies
??? pyproject.toml                 # Package configuration
?
??? docs/                          # Scientific & engineering documentation
?   ??? problem-statement.md       # KPIT Sparkle scope & agricultural dilemma
?   ??? solution-overview.md       # System concept & multi-modal intelligence
?   ??? architecture.md            # Component breakdown & data flows
?   ??? methodology.md             # Scientific methodology (existing, adapted, planned)
?   ??? migration-notes.md         # Technical migration log & audit notes
?
??? src/                           # Production source package
?   ??? preprocessing/             # 26D agronomic feature engineering (SFI, WSI, RD, SMI)
?   ??? geospatial/                # Soil, weather, elevation, and terrain acquisition
?   ??? models/                    # Crop inference, multi-objective ranking, yield regressor
?   ??? explainability/            # TreeSHAP & Groq LLM agronomic explainer
?   ??? vision/                    # Remote sensing & canopy segmentation specifications
?   ??? utils/                     # Configuration and shared utilities
?   ??? api/                       # FastAPI backend server
?
??? models/                        # Serialized ML artifacts & registries
?   ??? crop/                      # Versioned models (v1 through v5), metadata, model cards
?   ??? erosion_model.pkl          # Advisory environmental risk model
?
??? data/                          # Datasets and knowledge bases
?   ??? README.md                  # Dataset architecture, sources, and licensing
?   ??? agriculture/
?       ??? knowledge/v1/          # FAO EcoCrop & ICAR physiological rule database
?       ??? v1.0/                  # District coordinates, suitability, and yield records
?
??? artifacts/                     # Empirical validation reports, calibration curves, ablations
?   ??? crop_v2/
?   ??? crop_v3/
?   ??? crop_v4/
?   ??? crop_v5/
?
??? scripts/                       # Training, benchmarking, and auditing scripts
?   ??? build_crop_knowledge_base.py
?   ??? train_crop_v5.py
?   ??? experiments_crop_v2.py
?   ??? data/
?
??? tests/                         # Comprehensive automated pytest test suite
?   ??? test_api_endpoints.py
?   ??? test_crop_calibration.py
?   ??? test_crop_inference.py
?   ??? test_crop_model_registry.py
?   ??? test_crop_ranking.py
?   ??? test_crop_v2_model.py
?   ??? test_crop_v3_counterfactual.py
?   ??? test_crop_v3_knowledge.py
?   ??? test_crop_v3_ranking.py
?   ??? test_crop_v3_regression.py
?   ??? test_crop_v3_uncertainty.py
?   ??? test_crop_v4_validation.py
?   ??? test_crop_v5_validation.py
?   ??? test_erosion_regression.py
?   ??? test_feature_engineering.py
?   ??? test_soil_weather_terrain.py
?
??? ui/                            # Next.js 14 web dashboard
?   ??? app/
?   ??? components/
?   ??? server/
?   ??? package.json
?
??? notebooks/                     # Interactive research notebooks
??? assets/                        # Diagrams and visual documentation assets
```

---

## Installation

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Node.js 18+ and npm (for the web dashboard)
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/harshilsetty/GeoAgri-AI.git
cd GeoAgri-AI
```

### 2. Set Up Python Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. (Optional) Set Up UI Dashboard
```bash
cd ui
npm install
cd ..
```

### 4. Configuration
Create a `.env` file in the repository root (optional for basic local operation):
```bash
# Optional: Groq API Key for accelerated LLM explanations
GROQ_API_KEY=gsk_your_groq_api_key_here
GEO_AI_CROP_MODEL_VERSION=v5
```
*Note: If no Groq API key is provided, the system automatically falls back to deterministic, local rule-based explanations without failing.*

---

## Usage

### 1. Running the FastAPI Backend
```bash
uvicorn src.api.backend_api:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive OpenAPI Swagger documentation at: `http://localhost:8000/docs`

#### Sample API Request:
```bash
curl -X 'GET' \
  'http://localhost:8000/api/agriculture/recommend?lat=15.335&lon=76.46&season=Kharif&topK=5' \
  -H 'accept: application/json'
```

### 2. Python Programmatic Inference
```python
from src.models import agri_inference

# Generate full recommendation for Hampi region in Kharif season
recommendation = agri_inference.generate_crop_recommendation(
    lat=15.335,
    lon=76.46,
    season="Kharif",
    top_k=5,
    model_version="v5"
)

primary = recommendation["primary_recommendation"]
print(f"Top Recommended Crop: {primary['crop']}")
print(f"Suitability Score: {primary['suitability_percent']}%")
print(f"Expected Yield: {primary.get('predicted_yield_t_ha', 'N/A')} t/ha")
print(f"Key Positive Drivers: {primary['positive_drivers']}")
print(f"Limiting Factors: {primary['limiting_factors']}")
```

### 3. Running the UI Dashboard
```bash
cd ui
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## Dataset

* **Knowledge Base**: Curated physiological thresholds for 22 crops synthesizing FAO EcoCrop and ICAR baselines.
* **Geospatial Reference**: District centroid coordinate mapping for all Indian agricultural districts.
* **Historical Suitability & Yield**: Curated 26-feature records validating multi-objective ranking.
* For complete dataset schema, provenance, and validation reports, refer to [data/README.md](data/README.md).

---

## Model Information

* **Models**: Serialized in `models/crop/` (`v1` through `v5`).
* **Architecture**: XGBoost Multi-Class Classifier + Yield Regressor + Learning-to-Rank Ranker.
* **Calibration**: Formally audited across 10 bins with Brier score and ECE metric tracking.
* **Model Cards**: Each version includes a standardized `model_card.md` documenting hyperparameters, training hardware, and baseline performance.

---

## Explainability

GeoAgri AI implements a two-tier explainability framework:
1. **Local TreeSHAP Attribution**: Calculates exact Shapley feature values per inference, identifying which specific agronomic features (e.g., soil nitrogen, rainfall deviation, pH) drove the recommendation upward or downward.
2. **Contextual LLM Reasoning**: Feeds verifiable SHAP values and FAO EcoCrop boundaries into Groq LLaMA-3.1-8B-Instant with strict temperature ($T=0.2$) to produce concise, actionable advisory narratives with zero hallucination. If API access is unavailable, a certified deterministic rule-based local generator takes over.

---

## Evaluation

Model evaluation is documented in detail in `artifacts/crop_v5/FINAL_REPORT.md` and certified via automated tests:
* **Classification Accuracy**: Evaluated on independent test sets across multiple agro-climatic zones.
* **Ranking Metrics**: Precision@K, NDCG@K, and MRR@K audited for crop prioritization.
* **Uncertainty & Stability**: Decision margin metrics quantify confidence separation between top recommendations.

---

## Roadmap

* **Phase 1 (Completed)**: Core geospatial ingestion, 26D feature engineering, multi-class XGBoost models, FAO EcoCrop rule engine, TreeSHAP explainability, FastAPI backend, Next.js UI, repository migration.
* **Phase 2 (In Development)**: Sentinel-2 multispectral satellite ingestion, vegetation index computation (NDVI, NDWI), and canopy water stress estimation.
* **Phase 3 (Upcoming)**: Closed-loop FAO-56 daily water balance model, predictive irrigation advisory, and field boundary segmentation.
* **Phase 4 (Final)**: Pilot field trials, IoT ground-truth calibration, and KPIT Sparkle 2027 submission package.

---

## KPIT Sparkle 2027

GeoAgri AI is developed to compete in **KPIT Sparkle 2027**, targeting innovative AI-driven infrastructure and sustainability solutions for national and global impact.

* **Domain**: Smart Agriculture & Sustainable Resource Management
* **Focus Area**: AI for Crop Monitoring, Moisture Stress Detection & Irrigation Advisory

---

## Research / References

1. **FAO EcoCrop**: *Crop Environmental Requirements Database*, Food and Agriculture Organization of the United Nations.
2. **Allen, R. G., et al. (1998)**: *Crop Evapotranspiration - Guidelines for Computing Crop Water Requirements*, FAO Irrigation and Drainage Paper 56.
3. **ISRIC - World Soil Information**: *SoilGrids250m 2.0 - Global Gridded Soil Information*, 2020.
4. **Chen, T., & Guestrin, C. (2016)**: *XGBoost: A Scalable Tree Boosting System*, Proceedings of the 22nd ACM SIGKDD International Conference.
5. **Lundberg, S. M., et al. (2020)**: *From Local Explanations to Global Understanding with Explainable AI for Trees*, Nature Machine Intelligence.
6. **Indian Council of Agricultural Research (ICAR)**: *Zonal Soil Fertility Status and Agro-Climatic Advisories*, Ministry of Agriculture & Farmers Welfare, Government of India.

---

## Team

* **Harshil Somisetty** ? Project Lead & AI Software Engineer ([@harshilsetty](https://github.com/harshilsetty))
* Department of Computer Science & Engineering, Lovely Professional University (LPU)

---

## License

This project is licensed under the MIT License ? see the [LICENSE](LICENSE) file for details.
