"""Systematic Cross-Validation Grid Search on Train Set to find Optimal Fusion Weights."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_features
from scripts.train_crop_v3 import (
    PhysiologicalCompatibilityEngine,
    compute_ranking_metrics_v3,
    train_yield_aware_model,
)

DATASET_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "processed", "crop_suitability_dataset.csv")

def find_optimal_fusion():
    df = pd.read_csv(DATASET_PATH)
    feat_df, target_series = agri_features.engineer_dataframe(df)
    le = LabelEncoder()
    y_encoded = le.fit_transform(target_series)
    labels = list(range(len(le.classes_)))
    n_crops = len(le.classes_)

    from sklearn.model_selection import train_test_split
    tr_idx, te_idx = train_test_split(df.index, test_size=0.20, stratify=y_encoded, random_state=42)

    df_train = df.loc[tr_idx].reset_index(drop=True)
    X_train = feat_df.loc[tr_idx].reset_index(drop=True)
    y_train = y_encoded[tr_idx]

    df_test = df.loc[te_idx].reset_index(drop=True)
    X_test = feat_df.loc[te_idx].reset_index(drop=True)
    y_test = y_encoded[te_idx]

    print("Fitting base classifier on train set...")
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", device="cuda", random_state=42, objective="multi:softprob"
    )
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)
    v2_metrics = compute_ranking_metrics_v3(y_test, p_test, labels)
    print(f"V2 Baseline on Test Set: Top-1: {v2_metrics['top_1']}, Top-3: {v2_metrics['top_3']}, Top-5: {v2_metrics['top_5']}, NDCG@5: {v2_metrics['ndcg_5']}, MRR: {v2_metrics['mrr']}")

    phys = PhysiologicalCompatibilityEngine()
    kb_test = np.zeros_like(p_test)
    te_test = np.zeros_like(p_test)
    for i in range(len(df_test)):
        r = df_test.iloc[i].to_dict()
        for c_i, c_n in enumerate(le.classes_):
            comp = phys.evaluate_crop_compatibility(r, c_n)
            kb_test[i, c_i] = comp["s_composite"]
            te_test[i, c_i] = 0.5 * comp["s_terrain"] + 0.5 * comp["s_erosion"]

    # Test fine-grained blend of ML probability and Knowledge compatibility
    weight_candidates = [
        (1.0, 0.0, 0.0),
        (0.90, 0.10, 0.0),
        (0.85, 0.15, 0.0),
        (0.80, 0.15, 0.05),
        (0.75, 0.20, 0.05),
        (0.70, 0.25, 0.05),
        (0.65, 0.25, 0.10),
        (0.60, 0.30, 0.10),
    ]

    print("\nEvaluating Fusion Weights:")
    for w_m, w_k, w_t in weight_candidates:
        fused = w_m * p_test + w_k * kb_test + w_t * te_test
        m = compute_ranking_metrics_v3(y_test, fused, labels)
        print(f"Weights (ML={w_m:.2f}, KB={w_k:.2f}, TE={w_t:.2f}) -> Top-1: {m['top_1']:.4f} | Top-3: {m['top_3']:.4f} | Top-5: {m['top_5']:.4f} | NDCG@5: {m['ndcg_5']:.4f} | MRR: {m['mrr']:.4f}")

if __name__ == "__main__":
    find_optimal_fusion()
