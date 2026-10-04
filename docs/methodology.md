# ?? GeoAgri-AI Scientific Methodology

This document outlines the scientific methodologies incorporated into GeoAgri-AI, categorizing them into:
1. **Existing Methodologies** (inherited and verified from Geo AI foundation)
2. **Adapted Methodologies** (refined for agricultural decision-support)
3. **Future Methodologies** (planned for subsequent KPIT Sparkle 2027 milestones)

---

## 1. Existing Methodologies (Inherited & Verified)

### A. Geospatial Multi-Source Ingestion & Fusion
* Depth-stratified soil query querying ISRIC SoilGrids v2.0 for physical soil texture attributes (clay, sand, silt percentages, cation exchange capacity).
* Spatial coordinate mapping to representative agro-climatic zones across India to retrieve calibrated baseline chemical fertility values (available Nitrogen, Phosphorus, Potassium, Organic Carbon, Electrical Conductivity, pH) based on Indian Council of Agricultural Research (ICAR) zonal surveys.
* Retrieval of atmospheric temperature, relative humidity, historical precipitation aggregations (7d, 14d, 30d, 90d), 7-day precipitation forecasts, and FAO-56 Penman-Monteith reference evapotranspiration ($ET_0$) via Open-Meteo.
* Digital Elevation Model (DEM) slope extraction calculating local elevation differentials and terrain gradient.

### B. Machine Learning Modeling & Calibration
* Multi-class gradient boosted decision trees (XGBoost) trained on multi-district agro-climatic tabular records.
* Probability calibration evaluation using Brier score and Expected Calibration Error (ECE) across reliability curves.
* Environmental terrain vulnerability scoring via an 8-feature XGBoost erosion regression model used as an advisory risk penalty.

### C. TreeSHAP Attribution & Grounded LLM Explanations
* Computation of exact TreeSHAP values for top-ranked outputs, isolating positive feature drivers and negative limiting factors.
* Downstream prompt generation feeding strictly verified SHAP attributions into Groq LLaMA-3.1-8B-Instant with deterministic temperature ($T=0.2$) and rule-based local fallback.

---

## 2. Adapted Methodologies (Current Implementation)

### A. 26-Dimensional Agronomic Derived Feature Schema
Rather than feeding raw metrics directly to estimators, domain-specific agronomic indices are synthesized:
1. **Soil Fertility Index ($SFI \in [0, 100]$)**:
   $$SFI = 0.25 \cdot S_N + 0.25 \cdot S_P + 0.20 \cdot S_K + 0.15 \cdot S_{OC} + 0.15 \cdot S_{pH}$$
   Where $S_i$ represents normalized nutrient sufficiency functions.
2. **Water Stress Index ($WSI$)**:
   Ratio of atmospheric evaporative demand ($ET_0$) to available soil moisture and 30-day precipitation.
3. **Rainfall Deviation ($RD$)**:
   Percentage anomaly of recent 90-day precipitation relative to the long-term seasonal benchmark.
4. **Soil Moisture Index ($SMI$)**:
   Normalized rootzone moisture scaled relative to field capacity and wilting point.

### B. Multi-Objective Decision & Knowledge Fusion Engine
Predictions are ranked not solely by raw model probability, but through a multi-criteria decision function:
$$Score(crop) = w_1 P_{ML} + w_2 S_{Soil} + w_3 S_{Season} + w_4 S_{Climate} + w_5 (1 - R_{Erosion})$$
* **Physiological Hard Constraints**: Candidate crops undergo FAO EcoCrop physiological boundary checks. If local soil pH or temperature violates upper/lower lethal thresholds, the crop receives a severe compatibility penalty.
* **Yield-Aware Productivity Estimation**: Secondary XGBoost regressors estimate expected yield (t/ha) and relative yield potential compared to regional baselines.

---

## 3. Future Methodologies (Planned for KPIT Sparkle 2027)

### A. Multi-Spectral Remote Sensing & Canopy Indices `[Planned]`
* Computation of Crop Water Stress Index (CWSI) and Normalized Difference Water Index (NDWI) using Sentinel-2 Band 8A (Narrow NIR) and Band 11 (SWIR):
  $$NDWI = rac{ho_{NIR} - ho_{SWIR}}{ho_{NIR} + ho_{SWIR}}$$
* Automated deep learning field boundary delineation and canopy coverage segmentation using DeepLabV3+ with ResNet backbones.

### B. Closed-Loop Irrigation Scheduling Advisory `[Planned]`
* Dual crop coefficient ($K_c = K_{cb} + K_e$) water balance modeling following FAO-56 guidelines.
* Integration of weather forecasts to recommend daily irrigation volumes ($m^3/ha$) and schedule timing to avert upcoming moisture stress while avoiding over-watering.
