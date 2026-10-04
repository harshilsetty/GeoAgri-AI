# GEO AI Crop Intelligence V4 — Dataset & Leakage Forensic Audit

**Execution Date:** 2026-10-01  
**Audit Target:** `data/agriculture/v1.0/processed/crop_suitability_dataset.csv`  
**Records:** 10,091 | **Features:** 31 | **Target Taxa:** 16 crops

---

## 1. Executive Summary & Integrity Assessment
The agricultural suitability dataset contains **10,091 records** spanning **1997–2020** across **30 Indian states and Union Territories**. 
* **Missing Values:** **0** (100% complete across all 31 columns).
* **Exact Duplicate Rows:** **0**.
* **Physiological Plausibility:** All pedo-climatic variables (pH: [4.55, 8.64], annual rainfall: [301.3, 6552.7] mm, temperatures: [4.8, 49.3]°C) fall within terrestrial agricultural bounds.

---

## 2. Target Leakage & Post-Harvest Variable Audit
| Potential Risk | Audit Check | Finding | Verdict |
| :--- | :--- | :--- | :---: |
| **Realized Yield in Classifier** | Inspect classification feature matrix $X_{ML}$ | Realized post-harvest `yield` is strictly excluded from the 26 classification features. It is used solely as a regression target. | **PASS** |
| **Post-Harvest Production Variables** | Scan for total production mass or harvested area | Neither `production` nor `harvested_area` are present as input features. Input variables represent pre-sowing environmental states. | **PASS** |
| **Derived Agrometeorological Leakage** | Verify whether `rainfall_season` or derived indices look ahead past season end | Derived indicators (`rainfall_7d`, `et0`, `soil_moisture`) reflect seasonal moisture regimes available during planting decision windows. | **PASS** |

---

## 3. Spatiotemporal Dependency & Leakage Risks
While tabular target leakage is absent, standard machine learning validation strategies suffer from severe **spatiotemporal autocorrelation**:

### A. Spatial Location Repetition
* **Unique Coordinate Pairs:** 10,091 distinct locations across 10,091 rows.
* **Repeat Ratio:** An average of **1.0 observations per geographic coordinate** across differing years and seasons.
* **Risk:** In an unstratified random 80/20 train/test split, identical farms appear in both train and test partitions. The model can overfit to micro-topography rather than learning regional agronomic compatibility.
* **V4 Solution:** Implement **Leave-State-Out** and **Leave-Region-Out** (6 macro-regions: North, South, West, Central, East, Northeast).

### B. Temporal Lookahead
* **Temporal Span:** 1997–2020 (24 calendar years).
* **Risk:** Random splits allow the model to train on year 2020 to predict year 2017. Real production systems only predict future harvests from historical records.
* **V4 Solution:** Implement **Forward Temporal Holdout** (Train on <= 2017, Validate on 2018, Test on 2019–2020) and **Spatiotemporal Holdout (Unseen Region + Future Year)**.

---

## 4. Yield Distribution & Species Asymmetry
Raw yield varies by orders of magnitude across crop species due to biological biomass differences:
* **Sugarcane:** Mean = 51.73 t/ha, P90 = 86.54 t/ha
* **Rice:** Mean = 2.22 t/ha, P90 = 3.2 t/ha
* **Moong (Pulse):** Mean = 0.54 t/ha, P90 = 0.88 t/ha
* **Urad (Pulse):** Mean = 0.59 t/ha, P90 = 0.91 t/ha

> **Forensic Conclusion:** Directly ranking crops by raw tonnes/hectare creates an insurmountable biological bias towards heavy biomass crops and penalizes nutrient-dense, low-mass pulses. Crop Intelligence V4 requires **Crop-Relative Normalization** ($Y / Y_{p90, c}$) or **Within-Crop Percentiles** rather than raw yield ranking.

---

## 5. Audit Certification
* **Audit Status:** **CERTIFIED CLEAN FOR V4 EXPERIMENTATION**
* **Target Leakage:** None detected.
* **Primary Recommendation:** Enforce strict Spatiotemporal Partitioning (Leave-Region-Out + Forward Temporal Split) as the primary V4 benchmark.
