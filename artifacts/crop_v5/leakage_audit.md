# GEO AI Crop Intelligence V5 -- Forensic Leakage & Grouped Split Audit

## 1. Executive Summary
- **Decision Context Observations:** 10,091 queries
- **Candidate-Level Pairs:** 161,456 candidate instances (16 candidates per query)
- **Zero Query Fragmentation:** All 16 candidate items corresponding to a single decision context $q = (\text{lat}, \text{lon}, \text{year}, \text{season})$ strictly reside within the exact same data partition.
- **Pre-Harvest Predictor Invariant:** Realized harvest mass and yield are strictly excluded from ranking feature inputs.

## 2. Grouped Query Segregation
| Split Strategy | Grouping Variable | Group Count | Leakage Prevention Mechanism |
| :--- | :--- | :---: | :--- |
| **Random Split** | Query Index $q$ | 10,091 | 16-candidate blocks assigned atomically |
| **Leave-State-Out** | State | 30 | Complete geographic containment per fold |
| **Leave-Region-Out** | Macro-Region | 6 | Cross-state macro pedo-climatic isolation |
| **Forward Temporal** | Year | 24 | Historical data ($\le$ 2017) strictly predicts Future ($\ge$ 2019) |
| **Spatiotemporal Holdout** | Region + Year | Joint | Unseen Southern states in Future Years 2019-2020 |

## 3. Ground Truth Verification
- Target: Observed historical crop occurrence (Binary Relevance $y_{q, c} \in \{0, 1\}$) and Crop-Relative Yield Potential.
- No artificial ranking labels synthesized; compatibility scores are derived deterministically from FAO EcoCrop and ICAR agro-ecological standards.
