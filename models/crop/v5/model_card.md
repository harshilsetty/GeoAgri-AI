# GEO AI Model Card: Crop Intelligence V5.0

## Model Details
- **Model Name:** Crop Intelligence V5 (Multi-Objective Agronomic Decision Intelligence & Learning-to-Rank)
- **Version:** v5.0
- **Architecture:** Hybrid Candidate-Level LambdaMART (XGBRanker `rank:ndcg`) & Softmax Probabilistic Classifier + Multi-Objective Pareto Frontier Engine
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0, Driver 581.86)
- **Training Date:** October 2026

## Performance Benchmarks (V4 vs V5)
| Metric | V4 Baseline | V5 Promoted | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 19.05% | 18.86% | -0.19% |
| **Top-3 Accuracy** | 40.48% | 42.31% | +1.83% |
| **Top-5 Accuracy** | 59.34% | 58.24% | -1.10% |
| **NDCG@5** | 0.3873 | 0.3849 | -0.0024 |
| **MRR** | 0.3672 | 0.3689 | +0.0017 |
| **Macro F1** | 0.1643 | 0.1614 | -0.0029 |
| **Geographic Top-5 (LSO)** | 59.28% | 56.83% | -2.45% |
| **Temporal Top-5** | 56.41% | 58.97% | +2.56% |
| **Spatiotemporal Top-5** | 40.82% | 43.20% | +2.38% |
| **Yield R^2** | 0.954 | 0.954 | 0.000 |
| **Erosion Invariant** | 90.50% | 90.50% | 0.00% |

## Invariants & Compliance
- Erosion baseline maintained at exact 90.50% holdout accuracy.
- Zero data leakage across 10,091 grouped decision contexts.
- Backward compatibility guaranteed for all legacy endpoints.
