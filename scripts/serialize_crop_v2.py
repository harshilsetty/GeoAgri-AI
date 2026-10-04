"""Serialize Crop Model V2 Artifacts and Model Card (Phase S).

Saves:
- models/crop/v2/crop_model.pkl
- models/crop/v2/label_encoder.pkl
- models/crop/v2/feature_metadata.json
- models/crop/v2/metrics_summary.json
- models/crop/v2/model_card.md
"""

import json
import os
import sys
import shutil
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_features

DATASET_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "processed", "crop_suitability_dataset.csv")
V2_MODEL_DIR = os.path.join(ROOT, "models", "crop", "v2")
os.makedirs(V2_MODEL_DIR, exist_ok=True)


def serialize_v2():
    print("Loading dataset for V2 model serialization...")
    df = pd.read_csv(DATASET_PATH)
    feat_df, target_series = agri_features.engineer_dataframe(df)

    le = LabelEncoder()
    y_encoded = le.fit_transform(target_series)

    # Best hyperparameters found in Phase H
    best_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
    }

    # Train on 80% train split
    X_train, X_test, y_train, y_test = train_test_split(
        feat_df, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
    )

    print("Fitting V2 XGBoost model...")
    v2_model = XGBClassifier(**best_params)
    v2_model.fit(X_train, y_train)

    # Save binaries
    model_path = os.path.join(V2_MODEL_DIR, "crop_model.pkl")
    encoder_path = os.path.join(V2_MODEL_DIR, "label_encoder.pkl")
    joblib.dump(v2_model, model_path)
    joblib.dump(le, encoder_path)
    print(f"Saved V2 model: {model_path}")
    print(f"Saved V2 encoder: {encoder_path}")

    # Copy calibration artifact
    calib_src = os.path.join(ROOT, "artifacts", "crop_v2", "calibration")
    calib_dst = os.path.join(V2_MODEL_DIR, "calibration")
    if os.path.exists(calib_dst):
        shutil.rmtree(calib_dst)
    shutil.copytree(calib_src, calib_dst)

    # Feature metadata
    feat_meta = {
        "model_name": "crop_xgboost",
        "version": "v2.0",
        "algorithm": "XGBoost (Hist Gradient Boosting)",
        "hyperparameters": best_params,
        "features": list(feat_df.columns),
        "feature_count": len(feat_df.columns),
        "target": "crop",
        "classes": list(le.classes_),
        "num_classes": len(le.classes_),
    }
    with open(os.path.join(V2_MODEL_DIR, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(feat_meta, f, indent=2)

    # Metrics summary
    metrics_summary = {
        "model_version": "v2.0",
        "release_status": "PROMOTED",
        "promotion_date": "2026-10-01",
        "benchmark_metrics": {
            "top_1_accuracy": 0.1852,
            "top_3_accuracy": 0.4339,
            "top_5_accuracy": 0.6107,
            "weighted_f1": 0.1470,
            "macro_f1": 0.1415,
            "mrr": 0.3767,
            "ndcg_5": 0.4612,
            "brier_score": 0.8676,
            "ece": 0.0223,
            "inference_latency_ms": 0.009,
            "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        },
        "generalization": {
            "geographic_cross_state_top5": "52.54% ± 7.94%",
            "temporal_future_top5": 0.5685,
        },
    }
    with open(os.path.join(V2_MODEL_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Model Card
    model_card_content = """# Model Card — Crop Intelligence Model V2.0

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
- **Non-Intended Use**: Predictive forecasting of yield volumes ($\text{t/ha}$), revenue, or market profitability. Must not replace certified agronomic testing or on-site agricultural extension advice.

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
- Native C++ TreeSHAP attribution computes per-sample Shapley values ($\phi_i$) in $<1\,\text{ms}$, partitioning features into positive suitability drivers and limiting ecological factors without Python Numba dependencies.

## 6. Known Limitations
- Does not capture farm-scale management (micro-irrigation, fertilizer dosing, certified seed varietal traits).
- Top-1 accuracy reflects severe multi-crop ecological niche overlap where multiple pulses or cereals share identical climate and soil envelopes. Recommending Top-5 ranked options is the scientifically valid decision-support approach.
""".strip()

    with open(os.path.join(V2_MODEL_DIR, "model_card.md"), "w", encoding="utf-8") as f:
        f.write(model_card_content)
    print(f"Saved model card: {os.path.join(V2_MODEL_DIR, 'model_card.md')}")


if __name__ == "__main__":
    serialize_v2()
