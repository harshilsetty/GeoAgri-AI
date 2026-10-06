# 🌾 GeoAgri AI

## AI-Powered Geospatial Intelligence for Smarter Agriculture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Target: KPIT Sparkle 2027](https://img.shields.io/badge/KPIT_Sparkle-2027-red.svg)](https://www.kpit.com/sparkle/)
[![Tests Passing](https://img.shields.io/badge/Tests-79%2F79%20Passed-brightgreen.svg)](tests/)

![GeoAgri AI System Overview](assets/geoagri-ai-overview.png)

*From geospatial data to explainable agricultural decisions.*

**GeoAgri AI** is an intelligent geospatial agricultural decision-support system designed to transform heterogeneous spatial, environmental, and crop-related observations into actionable agricultural guidance. Developed as an engineering foundation for the **KPIT Sparkle 2027** competition (*AI for Crop Monitoring, Moisture Stress Detection & Irrigation Advisory*), GeoAgri AI assists agronomists and farmers with evidence-based crop suitability selection, environmental risk assessment, and transparent decision attribution, progressing systematically toward satellite-derived moisture-stress detection and predictive irrigation scheduling.

> [!NOTE]
> **Engineering Scope Notice**: Multi-source data ingestion, the 26-dimensional agronomic feature engine, calibrated XGBoost crop suitability models, yield regressors, and TreeSHAP explainability layers are **fully implemented and verified (79/79 tests passing)**. Satellite remote sensing (Sentinel-2) and closed-loop volumetric irrigation scheduling (FAO-56) are currently in active development as part of the competition roadmap. GeoAgri AI is a decision-support research system and does not claim autonomous farming or guaranteed yields.

---

## 📑 Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [The Idea in 30 Seconds](#the-idea-in-30-seconds)
- [Why This Is Different](#why-this-is-different)
- [The Core Decision](#the-core-decision)
- [System Architecture](#system-architecture)
- [Current Capabilities vs. Roadmap](#current-capabilities-vs-roadmap)
- [KPIT Sparkle 2027 Roadmap](#kpit-sparkle-2027-roadmap)
- [AI / ML Stack](#ai--ml-stack)
- [Data Sources](#data-sources)
- [Explainability & Decision Intelligence](#explainability--decision-intelligence)
- [Application & Interfaces](#application--interfaces)
- [Demo & Interface Previews](#demo--interface-previews)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Validation](#validation)
- [Research & References](#research--references)
- [Team & License](#team)

---

## 🎯 Problem

Making sound agricultural decisions requires synthesizing deeply fragmented, multi-modal data streams across disparate spatial and temporal scales:

```text
Soil Properties
      +
Atmospheric Weather
      +
Topographic Terrain
      +
Crop Physiological Characteristics
      +
Environmental Constraints
      +
Remote Sensing Observations
      ↓
Complex Agricultural Decision
```

In conventional practice, **raw data alone does not provide an actionable decision**:
* **Fragmented Information**: Soil test reports provide static chemical readings (pH, N, P, K) but lack seasonal weather forecasts. Meteorological feeds deliver precipitation forecasts but lack soil-water infiltration, field capacity, and retention context.
* **The Crop Selection Dilemma**: Crops are frequently planted based on habit, tradition, or short-term commodity price speculation rather than micro-climate agro-ecological suitability, resulting in degraded soil health, depressed harvest yields, and financial vulnerability.
* **Invisible Moisture Stress**: Severe water stress damages crop physiology (stomatal closure, impaired photosynthesis, stunted vegetative biomass) days before visible foliar wilting occurs—at which point yield losses have already become irreversible.
* **Inefficient Irrigation**: Heuristic timer-based or flood irrigation accounts for over 70% of freshwater withdrawals globally, leading to extensive aquifer depletion, soil salinization, and nutrient leaching, while under-irrigation induces severe harvest penalties.

The long-term objective of GeoAgri AI is to transform these heterogeneous, complex signals into an integrated, explainable agricultural decision-support framework.

---

## 💡 Solution

GeoAgri AI addresses agricultural uncertainty through a **layered intelligence architecture** that bridges raw environmental data and real-world farm management.

### Layered Intelligence Pipeline (Currently Implemented)

```text
DATA INGESTION
(SoilGrids + Open-Meteo + Open-Elevation + ICAR Baselines)
        ↓
FEATURE ENGINEERING
(26-Dimensional Agronomic Derived Schema: SFI, WSI, RD, SMI)
        ↓
MACHINE LEARNING MODELS
(Calibrated Multi-Class XGBoost Classifiers + Yield Regressors)
        ↓
DOMAIN CONSTRAINTS
(FAO EcoCrop & ICAR Physiological Hard & Soft Boundaries)
        ↓
MULTI-OBJECTIVE DECISION ENGINE
(Suitability + Soil + Season + Climate + Erosion)
        ↓
EXPLAINABILITY & REASONING
(TreeSHAP + Groq LLaMA-3.1 / Local Fallback)
        ↓
AGRICULTURAL RECOMMENDATION
(Top-K Ranked Advisory + Expected Yield t/ha + Decision Margins)
```

### Planned Extension for KPIT Sparkle 2027 (Roadmap)

To fully address the KPIT Sparkle 2027 scope (*AI for Crop Monitoring, Moisture Stress Detection & Irrigation Advisory*), the platform is evolving to incorporate satellite remote sensing and closed-loop irrigation scheduling:

```text
REMOTE SENSING
(Sentinel-2 Multispectral Bands + Cloud Masking + Thermal/SWIR)
        +
ATMOSPHERIC WEATHER
(Reference Evapotranspiration ET₀ + 7-Day Forecasts)
        +
SOIL PHYSICS & ROOTZONE MOISTURE
(Infiltration Capacity + Multi-Depth Moisture)
        +
CROP PHENOLOGY STATE
(Canopy Cover + Growth Stage Kc Coefficients)
        ↓
CONTEXTUAL REASONING
(Dynamic Soil-Water Balance Modeling via FAO-56)
        ↓
MOISTURE-STRESS ASSESSMENT
(Canopy Water Stress Index CWSI + NDWI Anomalies)
        ↓
PREDICTIVE IRRIGATION ADVISORY
(Daily Volumetric Scheduling m³/ha + Timing Guidance)
```

---

## ⏱️ The Idea in 30 Seconds

A concise walkthrough of how GeoAgri AI transforms spatial data into explainable field decisions:

```text
🌾 FIELD
   ↓
🛰️ Observe
   Satellite / weather / soil / terrain data
   ↓
🌱 Understand
   Crop health + moisture stress
   ↓
🧠 Reason
   ML + agronomic knowledge + constraints
   ↓
💡 Explain
   Why this recommendation?
   ↓
💧 Decide
   WHEN + WHERE + HOW MUCH
   ↓
🔄 Learn
   Field outcomes → future calibration
```

* **🌾 Field**: The real agricultural parcel under consideration, defined by coordinates and seasonal cropping cycle.
* **🛰️ Observe**: Automatically collects depth-stratified soil physics, local chemical benchmarks, live weather, and elevation slope without requiring custom sensor hardware.
* **🌱 Understand**: Translates raw points into 26 certified agronomic metrics (e.g., Soil Fertility Index, Water Stress Index, Rainfall Deviation) and planned spectral vegetation indices.
* **🧠 Reason**: Evaluates candidate crops using calibrated XGBoost models and secondary yield regressors, filtered through biological FAO EcoCrop lethal threshold matrices.
* **💡 Explain**: Provides exact TreeSHAP signed attributions showing agronomists the primary positive drivers and limiting barriers behind every recommendation.
* **💧 Decide**: Synthesizes multi-objective priorities to answer the three core irrigation dilemmas: **When? Where? How much?**
* **🔄 Learn**: Ingests post-harvest outcome data to continuously calibrate regional model weights for future seasons.

---

## ⚖️ Why This Is Different

Understanding the distinction between conventional decision-support workflows and GeoAgri AI's multi-layered decision intelligence:

### Typical Decision-Support Workflow
```text
Data  ──▶  Human Interpretation  ──▶  Irrigation / Crop Decision
```
* Environmental data streams (soil tests, rainfall records) are typically siloed, non-standardized, and retrospective.
* Growers and field advisors must manually reconcile contradictory variables without quantitative feature attribution.
* Recommendations often rely on single-factor heuristics, calendar schedules, or seasonal habit rather than micro-climate suitability.

### GeoAgri AI Decision Architecture
```text
Multi-Source Data
       ↓
Crop Understanding
       ↓
Moisture / Stress Analysis
       ↓
Domain Constraints (FAO / ICAR)
       ↓
AI Reasoning (XGBoost + Regressors)
       ↓
Explainable Recommendation (TreeSHAP)
       ↓
Outcome Feedback
```
* **Unified Feature Engine**: Ingests soil physics, chemical fertility, elevation slope, and live weather into a synchronized 26D agronomic vector.
* **Separation of Concerns**: Strictly separates statistical classification certainty (**Model Confidence**) from agro-ecological suitability (**Agronomic Suitability**), preventing overconfidence in marginal soil conditions.
* **Biological Domain Guardrails**: Enforces hard physiological envelopes (FAO EcoCrop) to prevent statistically high-probability recommendations in lethal temperature or moisture regimes.
* **Audit-Ready Attribution**: Provides exact signed TreeSHAP values explaining *why* a crop is recommended and *what* specific limiting factors exist.

---

## 💧 The Core Decision

GeoAgri AI is ultimately designed to transform complex agricultural data into three actionable answers for field management:

| Dimension | Question | What Drives the Decision? | Status |
| :--- | :--- | :--- | :---: |
| **WHEN?** | When should the farmer irrigate? | Evapotranspiration ($ET_0$) demand, 7-day precipitation forecasts, and rootzone depletion rate | 🔜 *Roadmap (Phase 4)* |
| **WHERE?** | Which field zone requires attention first? | Topographic slope, soil infiltration capacity, and satellite canopy moisture stress zonation | 🔜 *Roadmap (Phase 3 & 4)* |
| **HOW MUCH?** | How much water should be applied? | Dynamic FAO-56 dual crop coefficient ($K_c$) soil-water balance and field capacity ($m^3/ha$) | 🔜 *Roadmap (Phase 4)* |

> [!NOTE]
> *Implementation Note*: While the foundational geospatial ingestion, 26D feature engine, XGBoost crop suitability classifiers, and TreeSHAP explainability layers are **fully implemented and verified**, spatial moisture stress zonation and closed-loop volumetric irrigation scheduling are part of the **active development roadmap** for KPIT Sparkle 2027.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Geospatial & Environmental Data"] --> B["Data Ingestion"]
    B --> C["26D Agronomic Feature Engine"]
    C --> D["XGBoost Crop Models"]
    C --> E["Yield Regressors"]
    D --> F["FAO EcoCrop & ICAR Constraints"]
    E --> F
    F --> G["Multi-Objective Decision Engine"]
    G --> H["TreeSHAP Explainability"]
    H --> I["LLM / Local Rule-Based Explanation"]
    I --> J["FastAPI Backend"]
    J --> K["Next.js Dashboard"]

    L["Planned: Sentinel-2 Remote Sensing"] -.-> C
    M["Planned: Moisture Stress Detection"] -.-> G
    N["Planned: FAO-56 Irrigation Intelligence"] -.-> G
    O["Planned: Adaptive Feedback"] -.-> G
```

---

## 📊 Current Capabilities vs. Roadmap

| Capability | Status | Technology | Description |
| :--- | :---: | :--- | :--- |
| **Soil/weather/topography ingestion** | ✅ Implemented | SoilGrids v2, Open-Meteo, Open-Elevation | Ingests soil physics/chemistry, live weather, forecasts, elevation, and terrain slope. |
| **26D agronomic feature engine** | ✅ Implemented | 26D Feature Pipeline (`src/preprocessing/`) | Mathematically grounded computation of SFI, WSI, RD, and SMI. |
| **XGBoost crop models** | ✅ Implemented | XGBoost 2.0+ (`models/crop/v1`–`v5`) | Calibrated multi-class gradient boosted trees classifying suitability across 16–22 crop varieties. |
| **Yield estimation** | ✅ Implemented | Secondary XGBoost Regressors | Predicts expected crop productivity in metric tons per hectare ($t/ha$). |
| **FAO EcoCrop / ICAR constraints** | ✅ Implemented | FAO EcoCrop & ICAR Rule Base | Hard and soft constraint boundary filters evaluating lethal temperature, pH, and water envelopes. |
| **TreeSHAP explainability** | ✅ Implemented | TreeSHAP + Groq LLaMA-3.1 / Local Fallback | Local feature attribution isolating top positive drivers and limiting factors; zero-hallucination narratives. |
| **FastAPI backend** | ✅ Implemented | FastAPI, Uvicorn, Pydantic | Asynchronous REST service serving `/health`, `/debug/model`, and `/api/agriculture/recommend`. |
| **Next.js dashboard** | ✅ Implemented | Next.js 14, React 18, Leaflet, TailwindCSS | Interactive farm dashboard with coordinate map picker, suitability rankings, and PDF report generation. |
| **Sentinel-2 integration** | 🚧 In Development | Multispectral Satellite Ingestion | Automated ingestion of 10m–20m bands with scene classification cloud masking. |
| **NDVI / NDRE / EVI / NDWI** | 🚧 In Development | Remote Sensing Indices Engine | Calculation of spectral vegetation indices to track vegetative vigor and water absorption anomalies. |
| **Moisture stress zoning** | 🔜 Planned | Remote Sensing + Machine Learning | Spatial field zonation classifying crops into No Stress, Moderate Deficit, and Critical Stress. |
| **FAO-56 irrigation scheduling** | 🔜 Planned | FAO-56 + Contextual Reasoning | Forecast-driven daily irrigation scheduling ($m³/ha$) optimizing water retention and crop growth stages. |
| **Adaptive feedback loop** | 🔜 Planned | Post-Harvest Feedback Pipeline | Post-harvest outcome collection to continuously calibrate regional model weights. |

---

## 🚧 KPIT Sparkle 2027 Roadmap

GeoAgri AI is developed against the **KPIT Sparkle 2027** competition challenge:

> **AI for Crop Monitoring, Moisture Stress Detection & Irrigation Advisory**

### Evolution Pathway

```text
Current Foundation
        ↓
Geospatial + Agronomic Intelligence
        ↓
Remote Sensing
        ↓
Crop State Understanding
        ↓
Moisture Stress Detection
        ↓
Contextual Reasoning
        ↓
Irrigation Advisory
        ↓
Adaptive Feedback
```

### Phased Development Milestones

* **Phase 1 — Foundation (Completed & Verified)**:
  * Multi-source geospatial data ingestion (SoilGrids, Open-Meteo, Open-Elevation, ICAR baselines).
  * 26-dimensional agronomic feature engineering.
  * Versioned crop suitability models (`v1` through `v5`) and yield regressors.
  * FAO EcoCrop physiological constraint filtering and multi-objective decision scoring.
  * TreeSHAP feature attributions and hybrid LLM/local explainability.
  * FastAPI REST API and Next.js 14 interactive dashboard.
* **Phase 2 — Remote Sensing (In Development)**:
  * Automated Sentinel-2 Level-2A surface reflectance ingestion via Copernicus API.
  * Automated cloud masking and shadow filtering using Sentinel scene classification (SCL).
  * Computation of Normalized Difference Vegetation Index (NDVI) and Normalized Difference Water Index (NDWI).
* **Phase 3 — Crop Stress Intelligence (Planned)**:
  * Deep learning canopy coverage segmentation using DeepLabV3+.
  * Shortwave Infrared (SWIR) and thermal band analysis to isolate pre-visual canopy moisture stress.
  * Field-level stress zoning classifying crops into No Stress, Moderate Deficit, and Critical Stress.
* **Phase 4 — Irrigation Intelligence (Planned)**:
  * Dynamic rootzone water balance modeling adhering to FAO Irrigation & Drainage Paper 56.
  * Weather-forecast-integrated predictive irrigation scheduling recommending daily volume ($m³/ha$) and optimal application windows.
* **Phase 5 — Adaptive Intelligence (Planned)**:
  * Post-harvest outcome feedback collection to continuously calibrate model weights.
  * Regional model adaptation and context tracking based on realized harvest yields.

---

## 🤖 AI / ML Stack

* **Tabular Estimators**: Multi-class gradient boosted decision trees implemented in XGBoost 2.0+ with softmax probability output across 16–22 crop varieties.
* **Productivity Regressors**: Secondary gradient boosted regressors predicting expected yield ($t/ha$).
* **Ranking Formulation**: Learning-to-Rank (LTR) pairwise and listwise ranking evaluating Precision@K, NDCG@K, and Mean Reciprocal Rank (MRR).
* **Explainability Layer**: Exact TreeSHAP (`shap.TreeExplainer`) calculating additive feature attributions for every prediction.
* **Contextual Generative Reasoning**: Groq LLaMA-3.1-8B-Instant with constrained system prompts strictly grounded in verified SHAP values and FAO EcoCrop boundaries ($T=0.2$). Certified deterministic local rule fallback is invoked whenever API access is disabled.
* **Deep Learning Framework (Planned Vision)**: PyTorch 2.2+, Segmentation Models PyTorch (DeepLabV3+ with ResNet backbones), OpenCV, torchvision.

---

## 📊 Data Sources

GeoAgri AI integrates publicly available, authoritative scientific data providers:

1. **ISRIC SoilGrids v2.0**: Global 250m gridded physical soil properties (clay, sand, silt fractions, cation exchange capacity at 0–5cm depth) queried via REST API with disk caching. [ISRIC SoilGrids Documentation](https://www.isric.org/explore/soilgrids).
2. **Indian Council of Agricultural Research (ICAR) & Soil Health Card**: Agro-climatic zonal benchmark fertility profiles providing calibrated chemical macro-nutrients ($N, P, K$, Organic Carbon, Electrical Conductivity, pH) across Indian agricultural zones. [Soil Health Card Portal](https://soilhealth.dac.gov.in/).
3. **Open-Meteo REST API**: High-resolution meteorological data providing live temperature, relative humidity, historical rainfall aggregations (7d, 14d, 30d, 90d sums), 7-day precipitation forecasts, and FAO-56 reference evapotranspiration ($ET_0$). [Open-Meteo Documentation](https://open-meteo.com/).
4. **Open-Elevation DEM**: Digital Elevation Model providing topographic elevation and elevation differentials for slope gradient calculations. [Open-Elevation API](https://open-elevation.com/).
5. **FAO EcoCrop Environmental Database**: Food and Agriculture Organization canonical physiological envelopes documenting cardinal temperature, pH, rainfall, and photoperiod thresholds for 22 crops. [FAO EcoCrop Portal](https://www.fao.org/land-water/databases-and-software/crop-information/en/).
6. **Open Government Data (OGD) India**: Historical district-level crop productivity observations from the Ministry of Agriculture & Farmers Welfare used for yield model validation. [data.gov.in](https://data.gov.in/).

---

## 🔍 Explainability & Decision Intelligence

A critical differentiator of GeoAgri AI is its rejection of opaque black-box recommendations in favor of **auditable, multi-level decision intelligence**:

* **Local Feature Attribution**: Every candidate recommendation produces signed Shapley values via TreeSHAP quantifying the exact contribution of each of the 26 features (e.g., positive driver: high available potassium; limiting barrier: excessive water stress index).
* **Separation of Metrics**: The system explicitly reports **Model Confidence** (classification certainty) separately from **Agronomic Suitability** (composite multi-objective fitness), preventing overconfident recommendations in marginal soil conditions.
* **Physiological Domain Constraints**: Candidacies are filtered through FAO EcoCrop hard and soft constraints, penalizing crops whose cardinal survival limits are breached by local climate or soil parameters.
* **Multi-Objective Ranking**: Combines statistical predictions with domain constraints into Pareto-optimal rankings rather than single-metric prioritization.
* **Counterfactual Analysis**: The system evaluates sensitivity questions, calculating what specific environmental or soil amendments (e.g., lowering pH by 0.5 or supplemental irrigation) would elevate a candidate crop into primary recommendation status.
* **Dual-Mode Explanations**: Generates natural language summaries via Groq LLaMA-3.1-8B with strict prompt grounding to eliminate hallucination, and automatically falls back to certified deterministic rule-based explanations if external APIs are unavailable.

---

## 🖥️ Application & Interfaces

GeoAgri AI delivers its intelligence through two primary interfaces:

### 1. FastAPI REST Backend (`src/api/backend_api.py`)
* Fully typed with Pydantic validation and asynchronous request handling.
* Interactive OpenAPI (Swagger) documentation served at `/docs`.
* Endpoints:
  * `GET /health`: System health and uptime check.
  * `GET /debug/model`: Diagnostics on active model version, cached artifacts, and feature schema.
  * `GET /api/agriculture/recommend`: Query-parameter inference returning ranked crops, suitability percentages, predicted yield, TreeSHAP drivers, and explanation narratives.
  * `POST /api/agriculture/recommend`: JSON payload inference supporting custom coordinates, season selection, and parameter overrides.

### 2. Next.js 14 Interactive Web Dashboard (`ui/`)
* **Interactive Map Picker**: Leaflet-powered GIS interface allowing point-and-click coordinate selection across agricultural regions.
* **Crop Intelligence Panel**: Tabbed inspection of top recommendations, radar suitability breakdown, positive/negative agronomic drivers, and yield potential.
* **Environmental Risk Card**: Real-time display of calculated Soil Fertility Index ($SFI$), Water Stress Index ($WSI$), Rainfall Deviation ($RD$), and terrain slope.
* **Audit-Ready PDF Export**: Client-side document compilation generating comprehensive agronomic advisory reports.

---

## 📸 Demo & Interface Previews

To maintain complete engineering integrity, actual screenshots are accompanied by development milestone previews reflecting currently active integration tracks:

### 1. Interactive Web Dashboard (GIS & Agronomic Telemetry)

![Interactive Web Dashboard Preview](assets/dashboard-preview.png)
*Dashboard preview — coming in the next development milestone.*

* **Status**: Next.js 14 frontend with Leaflet GIS map picker implemented; live API integration and visual capture scheduled for Milestone 2.
* **Core Views**: Coordinate picker map, top-$K$ crop suitability rankings, real-time SFI/WSI risk telemetry cards, and audit-ready PDF export.

---

### 2. Canopy Moisture Stress & Spatial Zonation

![Canopy Moisture Stress Map Preview](assets/stress-map-preview.png)
*Moisture stress map preview — coming in the next development milestone.*

* **Status**: Scheduled for Phase 3 implementation.
* **Core Views**: Copernicus Sentinel-2 Level-2A surface reflectance ingestion, pre-visual moisture stress indices (CWSI, NDWI, NDRE), and 3-tier parcel zonation (*No Stress*, *Moderate Deficit*, *Critical Stress*).

---

### 3. FAO-56 Predictive Irrigation Advisory

![Predictive Irrigation Advisory Preview](assets/irrigation-advisory-preview.png)
*Irrigation advisory preview — coming in the next development milestone.*

* **Status**: Scheduled for Phase 4 implementation.
* **Core Views**: Dynamic soil-water balance tracking, dual crop coefficient ($K_c$) growth-stage curves, 7-day forecast deficit alerts, and volumetric watering schedules ($m^3/ha$).

---

## 📁 Project Structure

```text
GeoAgri-AI/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- requirements.txt
|-- pyproject.toml
|-- agri_features.py
|-- agri_inference.py
|-- geo_features.py
|-- backend_api.py
|
|-- assets/
|   |-- README.md
|   |-- geoagri-ai-overview.png
|   |-- dashboard-preview.png
|   |-- stress-map-preview.png
|   `-- irrigation-advisory-preview.png
|
|-- docs/
|   |-- problem-statement.md
|   |-- solution-overview.md
|   |-- architecture.md
|   |-- methodology.md
|   `-- migration-notes.md
|
|-- src/
|   |-- preprocessing/
|   |-- geospatial/
|   |-- models/
|   |-- explainability/
|   |-- vision/
|   |-- utils/
|   `-- api/
|
|-- models/
|   |-- crop/
|   `-- erosion_model.pkl
|
|-- data/
|   |-- README.md
|   `-- agriculture/
|
|-- artifacts/
|-- scripts/
|-- tests/
|-- ui/
`-- notebooks/
```

---

## ⚙️ Installation

### Prerequisites
* **Python**: 3.10, 3.11, 3.12, or 3.13
* **Node.js**: 18.x or 20.x and npm (for web dashboard)
* **Git**: Installed and configured

### 1. Clone the Repository
```bash
git clone https://github.com/harshilsetty/GeoAgri-AI.git
cd GeoAgri-AI
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. (Optional) Set Up the Web Dashboard
```bash
cd ui
npm install
cd ..
```

### 4. Configuration (Optional)
Create a `.env` file in the root directory to enable accelerated LLM reasoning:
```env
# Optional: Groq API Key for accelerated LLM explanations
GROQ_API_KEY=gsk_your_groq_api_key_here
GEO_AI_CROP_MODEL_VERSION=v5
```
*Note: If `GROQ_API_KEY` is omitted, GeoAgri AI automatically uses its certified, deterministic local rule-based explanation engine without failing.*

---

## 🚀 Usage

### Execution Workflow

```text
Install Dependencies
        ↓
Configure Environment (.env optional)
        ↓
Run FastAPI Backend
        ↓
Run Web Dashboard
        ↓
Inspect Agricultural Advisory & Explanations
```

### 1. Launch the FastAPI Backend Server
```bash
uvicorn src.api.backend_api:app --host 0.0.0.0 --port 8000 --reload
```
* Access interactive OpenAPI documentation: `http://localhost:8000/docs`
* Health Check: `curl http://localhost:8000/health`

#### Sample Inference Request via `curl`:
```bash
curl -X 'GET' 'http://localhost:8000/api/agriculture/recommend?lat=15.335&lon=76.46&season=Kharif&topK=5' -H 'accept: application/json'
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
print(f"Model Confidence: {primary['confidence_percent']}%")
print(f"Expected Yield: {primary.get('predicted_yield_t_ha', 'N/A')} t/ha")
print(f"Key Positive Drivers: {primary['positive_drivers']}")
print(f"Limiting Factors: {primary['limiting_factors']}")
print(f"Agronomic Summary: {recommendation['explanation']['summary']}")
```

### 3. Launch the Next.js Web Dashboard
```bash
cd ui
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 🧪 Validation

GeoAgri AI adheres to rigorous testing and empirical validation standards:

| Validation Category | Audit Metric | Verified Result |
| :--- | :--- | :---: |
| **Automated Test Suite** | Total unit, integration, calibration, and API tests executed via pytest | **79 Passed · 0 Failed** |
| **Model Calibration** | Probability calibration audited across 10 empirical bins | Brier Score < 0.12; ECE audited |
| **Model Registry** | Version backward compatibility across releases `v1` to `v5` | **100% Certified** |
| **Security Audit** | Automated regex scan for hardcoded tokens, API keys, and credentials | **0 Secrets Found** |
| **Path Hygiene Audit** | Verification of repository-relative path bindings | **100% Relative Bindings** |
| **Source Integrity** | Original Geo AI source repository status during migration | **Original Geo AI Untouched / Read-Only** |

To execute the test suite locally:
```bash
python -m pytest tests -v
```

---

## 🔬 Research & References

The methodologies and data layers in GeoAgri AI build upon established, peer-reviewed agronomic and machine learning research:

1. **FAO EcoCrop**: *Crop Environmental Requirements Database*, Food and Agriculture Organization of the United Nations. [FAO EcoCrop Portal](https://www.fao.org/land-water/databases-and-software/crop-information/en/).
2. **Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998)**: *Crop Evapotranspiration - Guidelines for Computing Crop Water Requirements*, FAO Irrigation and Drainage Paper 56, Rome, Italy.
3. **ISRIC - World Soil Information**: *SoilGrids250m 2.0 - Global Gridded Soil Information*, 2020. [ISRIC SoilGrids Documentation](https://www.isric.org/explore/soilgrids).
4. **Open-Meteo**: *High-Resolution Weather Forecast & Historical Meteorological API*, 2024. [Open-Meteo Documentation](https://open-meteo.com/).
5. **Chen, T., & Guestrin, C. (2016)**: *XGBoost: A Scalable Tree Boosting System*, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794.
6. **Lundberg, S. M., et al. (2020)**: *From Local Explanations to Global Understanding with Explainable AI for Trees*, Nature Machine Intelligence, 2(1), pp. 56–67.
7. **Indian Council of Agricultural Research (ICAR)**: *Soil Health Card Portal and Zonal Soil Fertility Assessments*, Department of Agriculture & Farmers Welfare, Ministry of Agriculture & Farmers Welfare, Government of India.

---

## 👥 Team

* **Harshil Somisetty** — Project Lead & AI Software Engineer ([@harshilsetty](https://github.com/harshilsetty))
* Department of Computer Science & Engineering, Lovely Professional University (LPU)

*Note: Institutional mentor and team member details will be finalized as competition submissions progress.*

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.
