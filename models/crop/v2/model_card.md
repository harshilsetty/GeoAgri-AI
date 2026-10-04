# Model Card — Crop Intelligence Model V2.0

## 1. Model Details
- **Model Name**: Geo AI Crop Intelligence Classifier
- **Model Version**: `v2.0`
- **Model Architecture**: Multi-Class Extreme Gradient Boosting (`xgboost.XGBClassifier`)
- **Framework**: XGBoost 3.2.0 / scikit-learn 1.6.1
- **Serialization Format**: Joblib Pickled Booster (`models/crop/v2/crop_model.pkl`)
- **Input Dimension**: 26 continuous and categorical agronomic features
- **Output Dimension**: Probability distribution across 16 crop classes

## 2. Intended Use & Disclaimers
- **Intended Purpose**: Decision-support exploratory tool evaluating the physiological and agro-climatic compatibility of Indian agricultural crops given local soil, climate, weather forecast, and terrain slope.
- **Non-Intended Use**: Predictive forecasting of yield volumes ($	ext{t/ha}$), revenue, or market profitability. Must not replace certified agronomic testing or on-site agricultural extension advice.

## 3. Training & Validation Methodology
- **Dataset**: `Indian Agricultural Crop Suitability Dataset (IACSD-v1.0)` (10,091 empirical records, 30 Indian states, 1997-2020).
- **Split Design**:
  - 80/20 Stratified Train/Test split for holdout benchmarking ($N = 2,019$ test records).
  - 3-Fold Train/Validation cross-validation for hyperparameter tuning.
  - Leave-State-Out grouped validation across 8 major agricultural states for geographic generalization.
  - Forward-chaining temporal validation (Train $\le 2015$, Test $2018-2020$) for temporal stability.

## 4. Evaluated Performance Metrics
| Metric | V1 Baseline | V2 Production | Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 18.08% | **18.52%** | +0.44% |
| **Top-3 Accuracy** | 41.80% | **43.39%** | +1.59% |
| **Top-5 Accuracy** | 58.89% | **61.07%** | **+2.18%** |
| **Mean Reciprocal Rank (MRR)** | 0.3520 | **0.3767** | +0.0247 |
| **Brier Score (Multi-class)** | 0.8832 | **0.8676** | **-0.0156 (Better)** |
| **Expected Calibration Error (ECE)** | 0.0450 | **0.0223** | **-0.0227 (Halved Error)** |
| **Geographic Cross-State Top-5** | 52.1% | **52.54% ± 7.94%** | Preserved |
| **Temporal Future Top-5 (2018-2020)** | 55.4% | **56.85%** | +1.45% |
| **CPU Latency per Query** | 0.016 ms | **0.009 ms** | **44% Faster** |
| **Model Size** | 2.8 MB | **1.1 MB** | **60% Lighter** |

## 5. Explainability
- Native C++ TreeSHAP attribution computes per-sample Shapley values ($\phi_i$) in $<1\,	ext{ms}$, partitioning features into positive suitability drivers and limiting ecological factors without Python Numba dependencies.

## 6. Known Limitations
- Does not capture farm-scale management (micro-irrigation, fertilizer dosing, certified seed varietal traits).
- Top-1 accuracy reflects severe multi-crop ecological niche overlap where multiple pulses or cereals share identical climate and soil envelopes. Recommending Top-5 ranked options is the scientifically valid decision-support approach.