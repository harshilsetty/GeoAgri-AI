# Crop Intelligence V3 Model Card

## Model Overview
- **Model Name:** Crop Intelligence V3 (Agronomic Ranking & Yield-Aware Decision Engine)
- **Version:** v3.0
- **Status:** PROMOTED
- **Training Device:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0)
- **Paradigm Shift:** Evolved from 16-class classification into an uncertainty-aware agronomic ranking system fusing learned probability, physiological compatibility, and expected yield potential.

## Architecture
1. **Base Agronomic Classifier:** Hist-XGBoost multiclass softprob model trained on 26 pedo-climatic features with CUDA acceleration.
2. **Physiological Compatibility Engine:** Grounded in FAO EcoCrop and ICAR handbooks, evaluating $S_{soil}$, $S_{climate}$, $S_{water}$, $S_{season}$, $S_{terrain}$, and $S_{erosion}$.
3. **Yield-Aware Regressor:** Hist-XGBoost Regressor predicting expected crop productivity $\hat{Y}(x, c)$ in t/ha with crop-specific P90 potential normalization.
4. **Ranking Objective:** Multi-system ranking optimizing NDCG@5, MRR, and decision stability.

## Benchmark Metrics (Test Set n=2019)
- **Top-1 Accuracy:** 19.02% (vs V2: 18.52%)
- **Top-3 Accuracy:** 43.59% (vs V2: 43.39%)
- **Top-5 Accuracy:** 61.32% (vs V2: 61.07%)
- **NDCG@3:** 0.3307 (vs V2: 0.3341)
- **NDCG@5:** 0.4032 (vs V2: 0.4612)
- **MRR:** 0.3801 (vs V2: 0.3767)
- **Expected Calibration Error (ECE):** 0.1180 (vs V2: 0.0223)
- **Geographic Cross-State Top-5:** 59.09% ± 0.71%
- **Temporal Future Top-5 (2018-2020):** 56.38%
- **Expected Yield RMSE:** 7.357 t/ha (R² = 0.745)
- **Inference Latency:** 0.036 ms/sample

## Scientific Grounding & Non-Fabrication
All physiological constraints are formally derived from the versioned agronomic knowledge base (`data/agriculture/knowledge/v1/crop_knowledge_base.json`). Expected yield is presented strictly as a statistical estimate with documented residual uncertainty.
