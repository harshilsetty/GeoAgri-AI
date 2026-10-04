# Crop Intelligence V4 Model Card

## Model Overview
- **Model Name:** Crop Intelligence V4 (Spatiotemporal Decision & Yield-Aware Ranking Engine)
- **Version:** v4.0
- **Status:** PROMOTED
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0)
- **Scientific Architecture:** 
  1. Base Classifier: XGBoost Softprob (26 pedo-climatic features, CUDA-accelerated)
  2. Grounded Knowledge Engine: FAO EcoCrop & ICAR Physiological Compatibility ($S_{soil}, S_{climate}, S_{water}, S_{season}, S_{terrain}, S_{erosion}$)
  3. Crop-Conditional Yield Model: Shared XGBoost regressor predicting $\hat{Y}(x, c)$ with calibrated uncertainty intervals (80%, 90%, 95%)
  4. Spatiotemporal Validation: Validated under Leave-State-Out, Leave-Region-Out, and Unseen Region + Future Year holdouts.

## Benchmark Performance Comparison (V3 vs V4)
| Metric | V3 Baseline | V4 Promoted | Change |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 19.02% | 19.05% | +0.03% |
| **Top-5 Accuracy** | 61.32% | 59.34% | -1.98% |
| **NDCG@5** | 0.4032 | 0.3873 | -0.0159 |
| **MRR** | 0.3801 | 0.3672 | -0.0129 |
| **Macro F1** | 0.1475 | 0.1643 | +0.0168 |
| **Geographic Top-5** | 59.09% | 59.28% | +0.19% |
| **Temporal Top-5** | 56.38% | 56.41% | +0.03% |
| **Spatiotemporal Top-5** | N/A | 40.82% | **New Ground Truth** |
| **Yield R^2** | 0.745 | 0.954 | +0.209 |
| **Inference Latency** | 0.036 ms | 0.038 ms | +0.002 ms |

## Verification
- **Erosion Model:** Immutability preserved (exact 90.50% holdout accuracy).
- **V1/V2/V3 Registries:** Preserved intact.
- **Leakage Status:** Fully audited, zero post-harvest target leakage.
