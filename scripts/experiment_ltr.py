"""Test Learning-to-Rank (XGBRanker with rank:ndcg and rank:pairwise) vs V2 Baseline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
import xgboost as xgb
import os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_features
from scripts.train_crop_v3 import PhysiologicalCompatibilityEngine, compute_ranking_metrics_v3

DATASET_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "processed", "crop_suitability_dataset.csv")

def run_ltr_experiment():
    df = pd.read_csv(DATASET_PATH)
    feat_df, target_series = agri_features.engineer_dataframe(df)
    le = LabelEncoder()
    y_encoded = le.fit_transform(target_series)
    labels = list(range(len(le.classes_)))
    n_crops = len(le.classes_)

    # 80/20 train/test split
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42, stratify=y_encoded)
    
    print(f"Building LTR dataset with {len(train_idx)} train queries and {len(test_idx)} test queries...")
    phys = PhysiologicalCompatibilityEngine()
    
    # Construct pairs (query, crop)
    def build_ltr_features(indices):
        X_rows = []
        y_rel = []
        groups = []
        for idx in indices:
            row_dict = df.loc[idx].to_dict()
            base_feats = feat_df.loc[idx].values
            true_crop = df.loc[idx, "crop"]
            
            for c_name in le.classes_:
                c_comp = phys.evaluate_crop_compatibility(row_dict, c_name)
                # Graded relevance
                if c_name == true_crop:
                    rel = 3
                elif c_comp["s_composite"] >= 0.85:
                    rel = 2
                elif c_comp["s_composite"] >= 0.65:
                    rel = 1
                else:
                    rel = 0
                
                # Crop specific features
                crop_onehot = [1.0 if c_name == c else 0.0 for c in le.classes_]
                comp_feats = [
                    c_comp["s_soil"], c_comp["s_climate"], c_comp["s_water"],
                    c_comp["s_season"], c_comp["s_terrain"], c_comp["s_erosion"], c_comp["s_composite"]
                ]
                row_vec = np.concatenate([base_feats, comp_feats, crop_onehot])
                X_rows.append(row_vec)
                y_rel.append(rel)
            groups.append(n_crops)
        return np.array(X_rows, dtype=np.float32), np.array(y_rel, dtype=np.int32), np.array(groups)

    X_train_ltr, y_train_ltr, groups_train = build_ltr_features(train_idx)
    X_test_ltr, y_test_ltr, groups_test = build_ltr_features(test_idx)
    
    print("Fitting XGBRanker with rank:ndcg on CUDA...")
    t0 = time.time()
    ranker = xgb.XGBRanker(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        objective="rank:ndcg",
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    ranker.fit(X_train_ltr, y_train_ltr, group=groups_train)
    fit_time = round(time.time() - t0, 3)
    print(f"LTR Ranker fitted in {fit_time}s")
    
    # Predict scores on test set
    preds_test = ranker.predict(X_test_ltr)
    score_matrix = preds_test.reshape(len(test_idx), n_crops)
    
    y_test_true = y_encoded[test_idx]
    ltr_metrics = compute_ranking_metrics_v3(y_test_true, score_matrix, labels)
    
    print("\n=== LTR XGBRanker Test Set Performance ===")
    for k, v in ltr_metrics.items():
        print(f"  {k}: {v}")
    
    # Compare with V2 Baseline
    v2_base = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, tree_method="hist", device="cuda", random_state=42, objective="multi:softprob"
    )
    v2_base.fit(feat_df.loc[train_idx], y_encoded[train_idx])
    v2_probs = v2_base.predict_proba(feat_df.loc[test_idx])
    v2_metrics = compute_ranking_metrics_v3(y_test_true, v2_probs, labels)
    
    print("\n=== V2 Softprob Classifier Test Set Performance ===")
    for k, v in v2_metrics.items():
        print(f"  {k}: {v}")

    # Also test an ensemble / fused score of LTR ranker score + V2 probability
    print("\n=== Blended Score (0.5 * V2_probs + 0.5 * LTR_score_normalized) ===")
    # min-max normalize LTR scores across classes
    ltr_norm = (score_matrix - score_matrix.min(axis=1, keepdims=True)) / np.maximum(1e-5, (score_matrix.max(axis=1, keepdims=True) - score_matrix.min(axis=1, keepdims=True)))
    blended = 0.60 * v2_probs + 0.40 * ltr_norm
    blend_metrics = compute_ranking_metrics_v3(y_test_true, blended, labels)
    for k, v in blend_metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_ltr_experiment()
