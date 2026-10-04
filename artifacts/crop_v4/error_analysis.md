# GEO AI Crop Intelligence V4 -- Error & Crop-Pair Confusion Analysis

## 1. Summary of Recommendation Errors
* **Total Evaluated Test Samples:** 546
* **Incorrect Top-1 Predictions:** 442 (80.95%)
* **True Crop in Top-3 (Near-Boundary Tie):** 117 (26.5% of errors)
* **True Crop in Top-5 (Ecologically Compatible):** 220 (49.8% of errors)

## 2. Error Breakdown by Category
- **Severe pedological divergence:** 204 samples (46.2%)
- **Agro-climatic niche overlap (True crop in Top-3):** 117 samples (26.5%)
- **Broad ecological compatibility (True crop in Top-5):** 103 samples (23.3%)
- **Cereal grain seasonal co-occurrence:** 11 samples (2.5%)
- **Legume/Pulse physiological similarity:** 7 samples (1.6%)

## 3. Top Confused Crop Pairs
| Crop A | Crop B | Mutual Confusion Count | Primary Agronomic Cause |
| :--- | :--- | :---: | :--- |
| **Moong** | **Rice** | 25 | Pulse family physiological convergence |
| **Maize** | **Rice** | 24 | High agro-climatic overlap & shared soil tolerance |
| **Groundnut** | **Rice** | 18 | High agro-climatic overlap & shared soil tolerance |
| **Pigeonpea** | **Rice** | 16 | High agro-climatic overlap & shared soil tolerance |
| **Chickpea** | **Wheat** | 15 | Rabi season Gangetic plain co-cultivation |
| **Rice** | **Sesamum** | 14 | High agro-climatic overlap & shared soil tolerance |
