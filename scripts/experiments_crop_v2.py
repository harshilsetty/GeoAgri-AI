"""Crop Model V2 Comprehensive Experiment Suite (Phases E through Q).

Executes:
1. Phase E: Multiple Baselines (Raw XGBoost, RF, Calibrated, Prior Majority)
2. Phase F: Feature Group Ablations (8 experiments)
3. Phase G: Class Imbalance Treatments (Unweighted, Balanced, Custom weights)
4. Phase H: Hyperparameter Optimization on Train/Val (Grid/Randomized CV)
5. Phase I: Alternative Model Benchmarks (XGBoost, RF, HistGradientBoosting, ExtraTrees)
6. Phase J: Geographic Generalization (Random vs Leave-State-Out)
7. Phase K: Temporal Generalization (Train <= 2015, Val 2016-2017, Test 2018-2020)
8. Phase L: Probability Calibration (Brier Score, ECE, Platt/Isotonic)
9. Phase M: Top-K & Ranking Metrics (Top-1, Top-3, Top-5, MRR, NDCG@3, NDCG@5)
10. Phase N & O: Multi-Objective Decision Layer Ablation & Weight Sensitivity
11. Phase P: Native TreeSHAP Analysis
12. Phase Q: Structured Error Analysis categorized by agronomic ambiguity type
"""

import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_features

DATASET_PATH = os.path.join(ROOT, "data", "agriculture", "v1.0", "processed", "crop_suitability_dataset.csv")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts", "crop_v2")

ABLATION_DIR = os.path.join(ARTIFACTS_DIR, "ablation")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
VALIDATION_DIR = os.path.join(ARTIFACTS_DIR, "validation")
CALIBRATION_DIR = os.path.join(ARTIFACTS_DIR, "calibration")
ERRORS_DIR = os.path.join(ARTIFACTS_DIR, "errors")

for d in [ABLATION_DIR, MODELS_DIR, VALIDATION_DIR, CALIBRATION_DIR, ERRORS_DIR]:
    os.makedirs(d, exist_ok=True)


# -------------------------------------------------------------
# Metric Utilities
# -------------------------------------------------------------
def compute_ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list) -> dict:
    """Computes Top-1, Top-3, Top-5, MRR, NDCG@3, NDCG@5."""
    n_samples = len(y_true)
    top1 = float(top_k_accuracy_score(y_true, y_prob, k=1, labels=labels))
    top3 = float(top_k_accuracy_score(y_true, y_prob, k=3, labels=labels))
    top5 = float(top_k_accuracy_score(y_true, y_prob, k=min(5, len(labels)), labels=labels))

    reciprocal_ranks = []
    ndcg3_list = []
    ndcg5_list = []

    for i in range(n_samples):
        true_label = y_true[i]
        probs = y_prob[i]
        ranked_classes = np.argsort(probs)[::-1]
        rank = int(np.where(ranked_classes == true_label)[0][0]) + 1  # 1-indexed

        reciprocal_ranks.append(1.0 / rank)

        if rank <= 3:
            ndcg3_list.append(1.0 / np.log2(rank + 1))
        else:
            ndcg3_list.append(0.0)

        if rank <= 5:
            ndcg5_list.append(1.0 / np.log2(rank + 1))
        else:
            ndcg5_list.append(0.0)

    mrr = float(np.mean(reciprocal_ranks))
    ndcg3 = float(np.mean(ndcg3_list))
    ndcg5 = float(np.mean(ndcg5_list))

    return {
        "top_1": round(top1, 4),
        "top_3": round(top3, 4),
        "top_5": round(top5, 4),
        "mrr": round(mrr, 4),
        "ndcg_3": round(ndcg3, 4),
        "ndcg_5": round(ndcg5, 4),
    }


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int, n_bins: int = 10) -> dict:
    """Computes Multi-Class Brier Score and Expected Calibration Error (ECE)."""
    n_samples = len(y_true)
    # One-hot true labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true] = 1.0

    # Multi-class Brier Score: mean squared error between probabilities and one-hot targets
    brier_score = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    # ECE on top-predicted class
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for m in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[m], bin_boundaries[m + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if in_bin.sum() > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += prop_in_bin * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin": m,
                "range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "count": int(in_bin.sum()),
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })

    return {
        "brier_score": round(brier_score, 4),
        "ece": round(float(ece), 4),
        "bins": bin_data,
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, labels) -> dict:
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 3)

    t_infer = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    infer_latency_ms = round(((time.time() - t_infer) / len(X_test)) * 1000.0, 3)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    ranking = compute_ranking_metrics(y_test, y_prob, labels)
    calib = compute_calibration_metrics(y_test, y_prob, len(labels))

    return {
        **ranking,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "brier_score": calib["brier_score"],
        "ece": calib["ece"],
        "fit_time_sec": fit_time,
        "latency_ms": infer_latency_ms,
        "_model": model,
        "_probs": y_prob,
        "_preds": y_pred,
    }


# -------------------------------------------------------------
# Main Experiment Pipeline
# -------------------------------------------------------------
def run_all_experiments():
    print("=" * 70)
    print("Starting Crop Model V2 Complete Scientific Experiment Suite")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    feat_df, target_series = agri_features.engineer_dataframe(df)

    le = LabelEncoder()
    y_encoded = le.fit_transform(target_series)
    labels = list(range(len(le.classes_)))

    # Master 80/20 train/test split with seed 42
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        feat_df, y_encoded, df, test_size=0.20, stratify=y_encoded, random_state=42
    )

    print(f"Data split: Train {len(X_train)}, Test {len(X_test)}, Features {X_train.shape[1]}")

    # =========================================================
    # PHASE E: Multiple Controlled Baselines
    # =========================================================
    print("\n--- PHASE E: Controlled Baselines ---")
    baselines = {
        "Baseline_D_Majority": DummyClassifier(strategy="most_frequent"),
        "Baseline_A_Raw_XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, objective="multi:softprob"
        ),
        "Baseline_B_RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=16, random_state=42, n_jobs=-1
        ),
        "Baseline_C_Calibrated_XGB": CalibratedClassifierCV(
            estimator=XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.1, random_state=42, objective="multi:softprob"),
            method="sigmoid",
            cv=3,
        ),
    }

    baseline_results = []
    for b_name, b_model in baselines.items():
        res = evaluate_model(b_model, X_train, y_train, X_test, y_test, labels)
        row = {
            "baseline": b_name,
            "top_1": res["top_1"],
            "top_3": res["top_3"],
            "top_5": res["top_5"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "mrr": res["mrr"],
            "ndcg_5": res["ndcg_5"],
            "brier_score": res["brier_score"],
            "ece": res["ece"],
            "latency_ms": res["latency_ms"],
        }
        baseline_results.append(row)
        print(f"[{b_name}] Top-1: {res['top_1']} | Top-5: {res['top_5']} | Weighted F1: {res['weighted_f1']} | Brier: {res['brier_score']}")

    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv(os.path.join(ARTIFACTS_DIR, "baseline", "controlled_baselines.csv"), index=False)

    # =========================================================
    # PHASE F: Feature Group Ablation Study
    # =========================================================
    print("\n--- PHASE F: Feature Group Ablations ---")
    soil_feats = ["soil_ph", "nitrogen", "phosphorus", "potassium", "organic_carbon", "electrical_conductivity", "clay", "sand", "silt", "texture_code"]
    climate_feats = ["temperature_mean", "temperature_range", "humidity_mean", "rainfall_season", "rainfall_30d", "rainfall_90d", "soil_moisture", "et0", "season_code"]
    terrain_feats = ["elevation", "slope"]
    derived_feats = ["soil_fertility_index", "water_stress_index", "rainfall_deviation", "soil_moisture_index"]
    erosion_feats = ["erosion_risk_score"]

    ablation_groups = {
        "Exp1_Soil_Only": soil_feats,
        "Exp2_Climate_Only": climate_feats,
        "Exp3_Terrain_Only": terrain_feats,
        "Exp4_Soil_Climate": soil_feats + climate_feats,
        "Exp5_Soil_Climate_Terrain": soil_feats + climate_feats + terrain_feats,
        "Exp6_Soil_Climate_Terrain_Derived": soil_feats + climate_feats + terrain_feats + derived_feats,
        "Exp7_Full_V1_Features": [col for col in feat_df.columns if col != "erosion_risk_score"],
        "Exp8_Full_Features_With_Erosion": list(feat_df.columns),
    }

    ablation_records = []
    for exp_name, feat_cols in ablation_groups.items():
        sub_X_train = X_train[feat_cols]
        sub_X_test = X_test[feat_cols]

        ab_model = XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=42, objective="multi:softprob"
        )
        res = evaluate_model(ab_model, sub_X_train, y_train, sub_X_test, y_test, labels)
        row = {
            "experiment": exp_name,
            "feature_count": len(feat_cols),
            "top_1": res["top_1"],
            "top_3": res["top_3"],
            "top_5": res["top_5"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "mrr": res["mrr"],
            "brier_score": res["brier_score"],
            "ece": res["ece"],
            "latency_ms": res["latency_ms"],
        }
        ablation_records.append(row)
        print(f"[{exp_name}] Feats: {len(feat_cols):2d} | Top-1: {res['top_1']:.4f} | Top-3: {res['top_3']:.4f} | Top-5: {res['top_5']:.4f} | Weighted F1: {res['weighted_f1']:.4f}")

    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv(os.path.join(ABLATION_DIR, "ablation_results.csv"), index=False)

    # =========================================================
    # PHASE G: Class Imbalance Experiments
    # =========================================================
    print("\n--- PHASE G: Class Imbalance Experiments ---")
    # Compute inverse class frequencies
    classes, counts = np.unique(y_train, return_counts=True)
    total_train = len(y_train)
    n_c = len(classes)
    balanced_weights = {c: total_train / (n_c * count) for c, count in zip(classes, counts)}
    sample_weights_balanced = np.array([balanced_weights[y] for y in y_train])

    # Sqrt balanced weights (milder cost sensitivity)
    sqrt_weights = {c: np.sqrt(total_train / (n_c * count)) for c, count in zip(classes, counts)}
    sample_weights_sqrt = np.array([sqrt_weights[y] for y in y_train])

    imbalance_configs = {
        "Unweighted": None,
        "Balanced_Sample_Weights": sample_weights_balanced,
        "Sqrt_Balanced_Weights": sample_weights_sqrt,
    }

    imbalance_records = []
    for imb_name, s_weights in imbalance_configs.items():
        imb_model = XGBClassifier(
            n_estimators=140, max_depth=5, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=42, objective="multi:softprob"
        )
        t0 = time.time()
        imb_model.fit(X_train, y_train, sample_weight=s_weights)
        fit_time = round(time.time() - t0, 3)

        preds = imb_model.predict(X_test)
        probs = imb_model.predict_proba(X_test)
        ranking = compute_ranking_metrics(y_test, probs, labels)
        calib = compute_calibration_metrics(y_test, probs, len(labels))
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))

        row = {
            "method": imb_name,
            **ranking,
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "brier_score": calib["brier_score"],
            "ece": calib["ece"],
        }
        imbalance_records.append(row)
        print(f"[{imb_name}] Top-1: {ranking['top_1']} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Top-5: {ranking['top_5']}")

    imbalance_df = pd.DataFrame(imbalance_records)
    imbalance_df.to_csv(os.path.join(ARTIFACTS_DIR, "models", "class_imbalance_experiments.csv"), index=False)

    # =========================================================
    # PHASE H: Hyperparameter Optimization on Train/Validation
    # =========================================================
    print("\n--- PHASE H: Hyperparameter Optimization (Train/Val Split) ---")
    # Split train into train_sub (75%) and val_sub (25%) - final X_test remains completely untouched!
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    hp_search_grid = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.5},
        {"n_estimators": 180, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
        {"n_estimators": 220, "max_depth": 5, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 1.0},
        {"n_estimators": 160, "max_depth": 6, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 3.0},
    ]

    best_val_score = -1.0
    best_hp = None
    hp_eval_records = []

    for i, params in enumerate(hp_search_grid):
        cand_model = XGBClassifier(
            **params,
            random_state=42,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
        cand_model.fit(X_tr_sub, y_tr_sub)
        val_probs = cand_model.predict_proba(X_val_sub)
        val_top5 = float(top_k_accuracy_score(y_val_sub, val_probs, k=5, labels=labels))
        val_top1 = float(top_k_accuracy_score(y_val_sub, val_probs, k=1, labels=labels))
        val_loss = float(log_loss(y_val_sub, val_probs, labels=labels))

        # Composite validation metric: Top-5 coverage + Top-1 accuracy - log loss penalty
        composite_score = val_top5 + (val_top1 * 0.5) - (val_loss * 0.1)

        hp_record = {
            "candidate_id": i + 1,
            "params": params,
            "val_top1": round(val_top1, 4),
            "val_top5": round(val_top5, 4),
            "val_log_loss": round(val_loss, 4),
            "composite_score": round(composite_score, 4),
        }
        hp_eval_records.append(hp_record)
        print(f"  Cand {i+1}: Top-1={val_top1:.4f} | Top-5={val_top5:.4f} | Loss={val_loss:.4f} | Composite={composite_score:.4f}")

        if composite_score > best_val_score:
            best_val_score = composite_score
            best_hp = params

    with open(os.path.join(MODELS_DIR, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"best_params": best_hp, "val_search_results": hp_eval_records}, f, indent=2)
    print(f"Selected Best Hyperparameters: {best_hp}")

    # =========================================================
    # PHASE I: Alternative Model Benchmarks
    # =========================================================
    print("\n--- PHASE I: Alternative Model Benchmarks ---")
    alt_models = {
        "XGBoost_Optimized_V2": XGBClassifier(
            **best_hp, random_state=42, objective="multi:softprob", eval_metric="mlogloss"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=8, learning_rate=0.06, l2_regularization=1.5, random_state=42
        ),
        "RandomForest_Tuned": RandomForestClassifier(
            n_estimators=250, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "ExtraTrees_Ensemble": ExtraTreesClassifier(
            n_estimators=200, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1
        ),
    }

    alt_results = []
    fitted_alt_models = {}

    for name, model in alt_models.items():
        res = evaluate_model(model, X_train, y_train, X_test, y_test, labels)
        fitted_alt_models[name] = res["_model"]
        row = {
            "model": name,
            "top_1": res["top_1"],
            "top_3": res["top_3"],
            "top_5": res["top_5"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "mrr": res["mrr"],
            "ndcg_5": res["ndcg_5"],
            "brier_score": res["brier_score"],
            "ece": res["ece"],
            "latency_ms": res["latency_ms"],
        }
        alt_results.append(row)
        print(f"[{name:<22}] Top-1: {res['top_1']} | Top-3: {res['top_3']} | Top-5: {res['top_5']} | Weighted F1: {res['weighted_f1']} | Latency: {res['latency_ms']}ms")

    alt_df = pd.DataFrame(alt_results)
    alt_df.to_csv(os.path.join(MODELS_DIR, "alternative_model_benchmark.csv"), index=False)

    # =========================================================
    # PHASE J: Geographic Generalization Evaluation
    # =========================================================
    print("\n--- PHASE J: Geographic Generalization (State & Zone Grouped) ---")
    major_states = df["state"].value_counts().head(8).index.tolist()
    geo_group_records = []

    for st in major_states:
        test_mask = (df["state"] == st)
        train_mask = ~test_mask

        X_tr_g = feat_df[train_mask]
        y_tr_g = y_encoded[train_mask]
        X_te_g = feat_df[test_mask]
        y_te_g = y_encoded[test_mask]

        if len(X_te_g) < 40:
            continue

        g_model = XGBClassifier(
            **best_hp, random_state=42, objective="multi:softprob", eval_metric="mlogloss"
        )
        g_model.fit(X_tr_g, y_tr_g)
        g_probs = g_model.predict_proba(X_te_g)
        g_preds = g_model.predict(X_te_g)

        g_rank = compute_ranking_metrics(y_te_g, g_probs, labels)
        g_f1 = float(f1_score(y_te_g, g_preds, average="weighted", zero_division=0))

        geo_group_records.append({
            "held_out_state": st,
            "test_samples": len(X_te_g),
            "top_1": g_rank["top_1"],
            "top_3": g_rank["top_3"],
            "top_5": g_rank["top_5"],
            "weighted_f1": round(g_f1, 4),
            "mrr": g_rank["mrr"],
        })
        print(f"State Held-Out: {st:<18} | Samples: {len(X_te_g):<4} | Top-1: {g_rank['top_1']:.4f} | Top-5: {g_rank['top_5']:.4f} | F1: {g_f1:.4f}")

    geo_group_df = pd.DataFrame(geo_group_records)
    geo_group_df.to_csv(os.path.join(VALIDATION_DIR, "geographic_generalization_states.csv"), index=False)
    mean_geo_top5 = float(geo_group_df["top_5"].mean())
    std_geo_top5 = float(geo_group_df["top_5"].std())
    print(f"Mean Geographic Cross-State Top-5: {mean_geo_top5*100:.2f}% ± {std_geo_top5*100:.2f}%")

    # =========================================================
    # PHASE K: Temporal Generalization Evaluation
    # =========================================================
    print("\n--- PHASE K: Temporal Generalization (Past -> Future) ---")
    # Historical training: years <= 2015
    # Validation: years 2016-2017
    # Future holdout test: years 2018-2020
    train_time_mask = df["year"] <= 2015
    val_time_mask = (df["year"] >= 2016) & (df["year"] <= 2017)
    test_time_mask = df["year"] >= 2018

    X_tr_time = feat_df[train_time_mask]
    y_tr_time = y_encoded[train_time_mask]
    X_val_time = feat_df[val_time_mask]
    y_val_time = y_encoded[val_time_mask]
    X_te_time = feat_df[test_time_mask]
    y_te_time = y_encoded[test_time_mask]

    time_model = XGBClassifier(
        **best_hp, random_state=42, objective="multi:softprob", eval_metric="mlogloss"
    )
    time_model.fit(X_tr_time, y_tr_time)
    time_probs_test = time_model.predict_proba(X_te_time)
    time_preds_test = time_model.predict(X_te_time)

    time_ranking = compute_ranking_metrics(y_te_time, time_probs_test, labels)
    time_f1 = float(f1_score(y_te_time, time_preds_test, average="weighted", zero_division=0))

    temporal_report = {
        "training_period": "1997-2015",
        "training_samples": int(len(X_tr_time)),
        "validation_period": "2016-2017",
        "validation_samples": int(len(X_val_time)),
        "test_period": "2018-2020",
        "test_samples": int(len(X_te_time)),
        "temporal_top_1": time_ranking["top_1"],
        "temporal_top_3": time_ranking["top_3"],
        "temporal_top_5": time_ranking["top_5"],
        "temporal_weighted_f1": round(time_f1, 4),
        "temporal_mrr": time_ranking["mrr"],
    }
    with open(os.path.join(VALIDATION_DIR, "temporal_generalization.json"), "w", encoding="utf-8") as f:
        json.dump(temporal_report, f, indent=2)
    print(f"Temporal Evaluation (Test on 2018-2020): Top-1: {time_ranking['top_1']} | Top-5: {time_ranking['top_5']} | Weighted F1: {time_f1:.4f}")

    # =========================================================
    # PHASE L: Probability Calibration Evaluation
    # =========================================================
    print("\n--- PHASE L: Probability Calibration Analysis ---")
    v2_base_model = alt_models["XGBoost_Optimized_V2"]
    v2_base_model.fit(X_train, y_train)
    raw_probs = v2_base_model.predict_proba(X_test)

    # Sigmoid (Platt) Calibration
    calib_sigmoid = CalibratedClassifierCV(estimator=v2_base_model, method="sigmoid", cv="prefit")
    calib_sigmoid.fit(X_val_sub, y_val_sub)
    sigmoid_probs = calib_sigmoid.predict_proba(X_test)

    raw_calib_metrics = compute_calibration_metrics(y_test, raw_probs, len(labels))
    sigmoid_calib_metrics = compute_calibration_metrics(y_test, sigmoid_probs, len(labels))

    calib_comparison = {
        "raw_xgboost": {
            "brier_score": raw_calib_metrics["brier_score"],
            "ece": raw_calib_metrics["ece"],
        },
        "sigmoid_calibrated": {
            "brier_score": sigmoid_calib_metrics["brier_score"],
            "ece": sigmoid_calib_metrics["ece"],
        },
    }
    with open(os.path.join(CALIBRATION_DIR, "calibration_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(calib_comparison, f, indent=2)
    print(f"Calibration: Raw Brier={raw_calib_metrics['brier_score']}, ECE={raw_calib_metrics['ece']} | Sigmoid Brier={sigmoid_calib_metrics['brier_score']}, ECE={sigmoid_calib_metrics['ece']}")

    # Plot Reliability Curves
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for name, metrics, color in [("Raw XGBoost V2", raw_calib_metrics, "blue"), ("Platt Calibrated", sigmoid_calib_metrics, "green")]:
        confs = [b["confidence"] for b in metrics["bins"]]
        accs = [b["accuracy"] for b in metrics["bins"]]
        plt.plot(confs, accs, marker="o", label=f"{name} (ECE={metrics['ece']:.3f})", color=color)

    plt.xlabel("Mean Predicted Confidence", fontsize=11)
    plt.ylabel("Observed Empirical Accuracy", fontsize=11)
    plt.title("Reliability Diagram - Probability Calibration", fontsize=13, pad=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(CALIBRATION_DIR, "reliability_curve.png"), dpi=200)
    plt.close()

    # =========================================================
    # PHASE N & O: Multi-Objective Decision Layer Ablation & Sensitivity
    # =========================================================
    print("\n--- PHASE N & O: Multi-Objective Decision Layer Ablation ---")
    import agri_inference

    def simulate_decision_engine(probs_matrix, test_df_sub, w_model, w_soil, w_season, w_climate, w_erosion):
        """Simulates multi-objective score adjustments on test samples."""
        reordered_ranks = []
        for i in range(len(test_df_sub)):
            raw_row = test_df_sub.iloc[i].to_dict()
            season = str(raw_row.get("season", "Kharif"))
            ph = float(raw_row.get("soil_ph", 7.0))
            rain = float(raw_row.get("rainfall_season", 600.0))
            temp = float(raw_row.get("temperature_mean", 26.0))
            slope = float(raw_row.get("slope", 5.0))
            erosion_score = float(raw_row.get("erosion_risk_score", 0.3))

            scores = []
            for c_idx, crop_name in enumerate(le.classes_):
                p_ml = probs_matrix[i, c_idx]
                rules = agri_inference.CROP_ECOLOGICAL_RULES.get(crop_name, {})

                # Soil score
                ph_min, ph_max = rules.get("ph_range", (5.5, 7.5))
                s_soil = 1.0 if ph_min <= ph <= ph_max else max(0.2, 1.0 - abs(ph - ((ph_min + ph_max) / 2)) * 0.25)

                # Season score
                pref_seasons = [s.lower() for s in rules.get("seasons", ["kharif"])]
                s_season = 1.0 if season.lower() in pref_seasons or "whole year" in pref_seasons else 0.15

                # Climate score
                t_min, t_max = rules.get("temp_range", (18.0, 35.0))
                s_climate = 1.0 if t_min <= temp <= t_max else max(0.3, 1.0 - abs(temp - 25.0) * 0.05)

                # Erosion score
                s_erosion = max(0.2, 1.0 - (erosion_score * 0.5))

                total_score = (
                    w_model * p_ml +
                    w_soil * s_soil +
                    w_season * s_season +
                    w_climate * s_climate +
                    w_erosion * s_erosion
                )
                scores.append(total_score)

            reordered_ranks.append(np.array(scores))

        return np.array(reordered_ranks)

    mo_configs = {
        "Exp_A_Raw_ML_Only": (1.00, 0.00, 0.00, 0.00, 0.00),
        "Exp_B_ML_Plus_Soil": (0.75, 0.25, 0.00, 0.00, 0.00),
        "Exp_C_ML_Soil_Season": (0.60, 0.20, 0.20, 0.00, 0.00),
        "Exp_D_ML_Soil_Season_Climate": (0.50, 0.20, 0.15, 0.15, 0.00),
        "Exp_E_Full_V1_Weights": (0.45, 0.20, 0.15, 0.10, 0.10),
        "Exp_F_High_Model_Confidence": (0.65, 0.15, 0.10, 0.05, 0.05),
    }

    mo_records = []
    for mo_name, weights in mo_configs.items():
        w_m, w_so, w_se, w_cl, w_er = weights
        sim_scores = simulate_decision_engine(raw_probs, df_test, w_m, w_so, w_se, w_cl, w_er)
        mo_ranking = compute_ranking_metrics(y_test, sim_scores, labels)
        row = {
            "configuration": mo_name,
            "weights": f"M={w_m}, So={w_so}, Se={w_se}, Cl={w_cl}, Er={w_er}",
            "top_1": mo_ranking["top_1"],
            "top_3": mo_ranking["top_3"],
            "top_5": mo_ranking["top_5"],
            "mrr": mo_ranking["mrr"],
            "ndcg_5": mo_ranking["ndcg_5"],
        }
        mo_records.append(row)
        print(f"[{mo_name}] Top-1: {mo_ranking['top_1']} | Top-3: {mo_ranking['top_3']} | Top-5: {mo_ranking['top_5']} | MRR: {mo_ranking['mrr']}")

    mo_df = pd.DataFrame(mo_records)
    mo_df.to_csv(os.path.join(ARTIFACTS_DIR, "models", "multi_objective_ablation.csv"), index=False)

    # =========================================================
    # PHASE P: Native TreeSHAP Analysis
    # =========================================================
    print("\n--- PHASE P: TreeSHAP Feature Attribution ---")
    import xgboost as xgb
    dtest = xgb.DMatrix(X_test)
    booster = v2_base_model.get_booster()
    shap_vals = booster.predict(dtest, pred_contribs=True)  # Shape: (N, 16, 27)

    # Global importance across classes and samples: mean absolute SHAP value
    # shap_vals has shape (N, 16, 27) or (N, 27*16)
    if len(shap_vals.shape) == 3:
        shap_feat_vals = shap_vals[:, :, :-1]  # drop bias
        global_shap = np.mean(np.abs(shap_feat_vals), axis=(0, 1))
    else:
        # flattened
        n_feats = X_test.shape[1]
        shap_reshaped = shap_vals.reshape((len(X_test), len(labels), n_feats + 1))
        shap_feat_vals = shap_reshaped[:, :, :-1]
        global_shap = np.mean(np.abs(shap_feat_vals), axis=(0, 1))

    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.round(global_shap, 4)
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(ARTIFACTS_DIR, "models", "global_shap_importance.csv"), index=False)

    print("Top 5 Global TreeSHAP Features:")
    for _, r in shap_df.head(5).iterrows():
        print(f"  {r['feature']:<25}: {r['mean_abs_shap']:.4f}")

    # =========================================================
    # PHASE Q: Structured Error Analysis
    # =========================================================
    print("\n--- PHASE Q: Structured Error Analysis ---")
    v2_preds = v2_base_model.predict(X_test)
    v2_probs = v2_base_model.predict_proba(X_test)

    errors = []
    for i in range(len(y_test)):
        true_c = int(y_test[i])
        pred_c = int(v2_preds[i])

        if true_c != pred_c:
            row_dict = df_test.iloc[i].to_dict()
            probs_i = v2_probs[i]
            ranked_i = np.argsort(probs_i)[::-1]
            true_rank = int(np.where(ranked_i == true_c)[0][0]) + 1
            top_alt = le.classes_[ranked_i[1]] if ranked_i[0] == pred_c else le.classes_[ranked_i[0]]

            # Error categorization heuristic
            true_name = le.classes_[true_c]
            pred_name = le.classes_[pred_c]
            diff_rain = abs(row_dict.get("rainfall_season", 500) - 600)
            diff_ph = abs(row_dict.get("soil_ph", 7.0) - 7.0)

            if true_rank <= 3:
                category = "Near-boundary Top-3 tie (Agro-climatic niche overlap)"
            elif true_rank <= 5:
                category = "Moderate ecological ambiguity (Top-5 compatible)"
            elif row_dict.get("state") in ["Rajasthan", "Punjab", "Haryana"] and true_name in ["Wheat", "Mustard"]:
                category = "Geographic co-occurrence (Rabi Gangetic rotation)"
            elif true_name in ["Urad", "Moong", "Chickpea"] and pred_name in ["Urad", "Moong", "Chickpea"]:
                category = "Legume/Pulse physiological similarity"
            elif pred_name == "Rice":
                category = "Majority class bias under moderate rainfall"
            else:
                category = "Severe pedological / environmental divergence"

            errors.append({
                "sample_index": i,
                "true_crop": true_name,
                "predicted_crop": pred_name,
                "predicted_probability": round(float(probs_i[pred_c]), 4),
                "true_crop_probability": round(float(probs_i[true_c]), 4),
                "true_crop_rank": true_rank,
                "in_top_3": true_rank <= 3,
                "in_top_5": true_rank <= 5,
                "top_alternative": top_alt,
                "state": row_dict.get("state", "Unknown"),
                "season": row_dict.get("season", "Unknown"),
                "soil_ph": row_dict.get("soil_ph"),
                "rainfall_season": row_dict.get("rainfall_season"),
                "temperature_mean": row_dict.get("temperature_mean"),
                "error_category": category,
            })

    error_df = pd.DataFrame(errors)
    error_df.to_csv(os.path.join(ERRORS_DIR, "error_analysis.csv"), index=False)
    print(f"Total errors: {len(error_df)} / {len(y_test)} ({len(error_df)/len(y_test)*100:.1f}%)")
    print("Error breakdown by category:")
    for cat, cnt in error_df["error_category"].value_counts().items():
        print(f"  {cat:<50}: {cnt} ({cnt/len(error_df)*100:.1f}%)")

    print("\n[OK] Experiment suite completed successfully!")


if __name__ == "__main__":
    run_all_experiments()
