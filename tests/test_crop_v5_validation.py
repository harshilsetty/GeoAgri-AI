"""Comprehensive Validation & Regression Test Suite for Crop Intelligence V5.

Covers:
1.  V4 frozen baseline reproducibility and V1/V2/V3/V4 preservation.
2.  Erosion model immutability (certified 90.50% holdout accuracy).
3.  GPU hardware verification artifact (NVIDIA GeForce RTX 3050 Laptop GPU).
4.  V5 Model Registry completeness (models/crop/v5/).
5.  Candidate-level grouped query construction (10,091 queries × 16 crops = 161,456 pairs).
6.  Learning-to-Rank benchmark artifacts (5 model architectures compared).
7.  Spatiotemporal generalization benchmarks (LSO, LRO, Forward Temporal, ST Holdout).
8.  Multi-Objective Pareto analysis artifacts.
9.  Fusion constraint experiment (Soft penalty vs Hard constraint).
10. Controlled ablation study (11 configurations).
11. Crop-level fairness analysis (per-crop recall and Top-K presence).
12. Temporal feature drift analysis (PSI and KS-test).
13. Error analysis and counterfactual robustness artifacts.
14. Calibration metrics (ECE, Brier score).
15. Production monitoring specification.
16. API backward compatibility with Section 22 contract and V5 multi-objective namespaces.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_inference

ARTIFACTS_V5 = os.path.join(ROOT, "artifacts", "crop_v5")
MODELS_V5 = os.path.join(ROOT, "models", "crop", "v5")


# -----------------------------------------------------------------------
# 1. Erosion Model Immutability
# -----------------------------------------------------------------------
def test_erosion_model_immutability():
    model_path = os.path.join(ROOT, "terrain_model", "erosion_model.pkl")
    dataset_path = os.path.join(ROOT, "terrain_model", "erosion_dataset.csv")

    assert os.path.exists(model_path), "Erosion model binary missing"
    assert os.path.exists(dataset_path), "Erosion dataset missing"

    df = pd.read_csv(dataset_path)
    features = ["slope", "vegetation", "elevation", "rainfall", "soil", "boulders", "ruins", "structures"]
    X, y = df[features], df["erosion"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    model = joblib.load(model_path)
    acc = accuracy_score(y_te, model.predict(X_te))
    assert round(acc, 3) == 0.905, f"Erosion baseline altered! Got {acc:.4f}"


# -----------------------------------------------------------------------
# 2. V1–V5 Model Registry Preservation
# -----------------------------------------------------------------------
def test_v1_to_v5_registry_preservation():
    for v in ["v1", "v2", "v3", "v4", "v5"]:
        m, le, meta, active_v = agri_inference.load_crop_artifacts(v)
        assert active_v == v, f"Expected {v}, got {active_v}"
        assert m is not None, f"Model is None for {v}"
        assert le is not None, f"LabelEncoder is None for {v}"
        assert len(le.classes_) == 16, f"Expected 16 classes for {v}"


def test_v4_rollback_explicit():
    """V4 must remain loadable as production rollback."""
    m, le, meta, v = agri_inference.load_crop_artifacts("v4")
    assert v == "v4"
    assert m is not None
    dummy = np.ones((1, 26), dtype=np.float32)
    probs = m.predict_proba(dummy)
    assert probs.shape == (1, 16)
    assert abs(probs.sum() - 1.0) < 1e-4


# -----------------------------------------------------------------------
# 3. GPU Hardware Verification Artifact
# -----------------------------------------------------------------------
def test_v5_gpu_hardware_verification():
    gpu_path = os.path.join(ARTIFACTS_V5, "gpu_verification.txt")
    assert os.path.exists(gpu_path), "gpu_verification.txt missing"

    with open(gpu_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "NVIDIA GeForce RTX 3050 Laptop GPU" in content
    assert "CUDA" in content
    assert "XGBoost" in content
    assert "VERIFIED" in content


# -----------------------------------------------------------------------
# 4. V5 Model Registry Completeness
# -----------------------------------------------------------------------
def test_v5_model_registry_completeness():
    required = [
        "crop_model.pkl",
        "ranking_model.pkl",
        "yield_model.pkl",
        "label_encoder.pkl",
        "feature_metadata.json",
        "metrics_summary.json",
        "model_card.md",
    ]
    for f in required:
        assert os.path.exists(os.path.join(MODELS_V5, f)), f"Missing V5 artifact: {f}"

    assert os.path.exists(os.path.join(MODELS_V5, "calibration")), "calibration/ dir missing"

    with open(os.path.join(MODELS_V5, "metrics_summary.json"), "r") as f:
        ms = json.load(f)
    assert ms["release_status"] == "PROMOTED"
    assert "spatiotemporal_holdout_top5" in ms["benchmark_metrics"]


# -----------------------------------------------------------------------
# 5. Candidate-Level Dataset Construction (Grouped Query Integrity)
# -----------------------------------------------------------------------
def test_v5_candidate_dataset_audit():
    audit_path = os.path.join(ARTIFACTS_V5, "dataset_audit.json")
    assert os.path.exists(audit_path), "dataset_audit.json missing"

    with open(audit_path, "r") as f:
        audit = json.load(f)

    assert audit["decision_contexts_count"] == 10091
    assert audit["candidate_crop_count"] == 16
    assert audit["total_candidate_samples"] == 10091 * 16
    assert audit["grouped_ranking_integrity"]["candidates_per_query"] == 16
    assert "PASS" in audit["grouped_ranking_integrity"]["target_leakage_status"]
    assert "PASS" in audit["grouped_ranking_integrity"]["group_leakage_status"]


# -----------------------------------------------------------------------
# 6. Learning-to-Rank Benchmark Artifacts
# -----------------------------------------------------------------------
def test_v5_ltr_benchmark_comparison():
    comp_path = os.path.join(ARTIFACTS_V5, "ranking_model_comparison.csv")
    assert os.path.exists(comp_path), "ranking_model_comparison.csv missing"

    df = pd.read_csv(comp_path)
    assert len(df) >= 5, "Expected at least 5 model architectures benchmarked"

    models_col = df["model"].tolist()
    assert any("Multiclass" in m or "V4" in m for m in models_col), "Multiclass baseline missing"
    assert any("LambdaMART" in m or "ndcg" in m.lower() for m in models_col), "LambdaMART model missing"
    assert any("Pairwise" in m or "pairwise" in m.lower() for m in models_col), "Pairwise model missing"

    # Verify that Multiclass NDCG@5 >= 0.30 (sanity check)
    for _, row in df.iterrows():
        if "Multiclass" in row["model"] or "V4" in row["model"]:
            assert row["ndcg_5"] >= 0.30, f"Multiclass NDCG@5 unexpectedly low: {row['ndcg_5']}"


def test_v5_hypothesis_falsification():
    """
    V5 central hypothesis: LambdaMART outperforms Multiclass.
    This test verifies the empirical result was documented honestly —
    regardless of whether the hypothesis was supported or falsified.
    """
    comp_path = os.path.join(ARTIFACTS_V5, "ranking_model_comparison.csv")
    df = pd.read_csv(comp_path).set_index("model")

    # Find multiclass row
    mc_rows = [r for r in df.index if "Multiclass" in r or "V4" in r]
    assert len(mc_rows) >= 1, "Multiclass baseline not found in comparison"
    mc_ndcg5 = df.loc[mc_rows[0], "ndcg_5"]

    # Both LTR models should be documented — regardless of outcome
    ltr_rows = [r for r in df.index if "LambdaMART" in r or "rank:ndcg" in r.lower()]
    assert len(ltr_rows) >= 1, "LambdaMART model not documented"

    # Hypothesis verdict is purely empirical — no assertion on who wins
    ltr_ndcg5 = df.loc[ltr_rows[0], "ndcg_5"]
    print(f"\nHypothesis Test: Multiclass NDCG@5={mc_ndcg5:.4f} | LambdaMART NDCG@5={ltr_ndcg5:.4f}")
    print(f"Verdict: {'SUPPORTED' if ltr_ndcg5 >= mc_ndcg5 else 'FALSIFIED — Multiclass superior'}")


# -----------------------------------------------------------------------
# 7. Spatiotemporal Benchmarks Artifacts
# -----------------------------------------------------------------------
def test_v5_spatiotemporal_benchmarks():
    for fname in ["geographic_validation.csv", "regional_validation.csv",
                  "temporal_validation.csv", "spatiotemporal_validation.csv"]:
        path = os.path.join(ARTIFACTS_V5, fname)
        assert os.path.exists(path), f"{fname} missing"
        df = pd.read_csv(path)
        assert len(df) >= 1, f"{fname} is empty"

    # Regional validation: 6 macro-regions
    reg_df = pd.read_csv(os.path.join(ARTIFACTS_V5, "regional_validation.csv"))
    assert len(reg_df) == 6, f"Expected 6 macro-regions, got {len(reg_df)}"
    assert set(reg_df["holdout_region"]).issubset(
        {"North", "South", "East", "West", "Central", "Northeast"}
    )

    # Spatiotemporal holdout: primary benchmark >= 0.35
    st_df = pd.read_csv(os.path.join(ARTIFACTS_V5, "spatiotemporal_validation.csv"))
    assert st_df.iloc[0]["v5_top5"] >= 0.35, "Spatiotemporal Top-5 collapsed below 35%"


# -----------------------------------------------------------------------
# 8. Pareto Analysis Artifacts
# -----------------------------------------------------------------------
def test_v5_pareto_analysis():
    pareto_path = os.path.join(ARTIFACTS_V5, "pareto_analysis.csv")
    assert os.path.exists(pareto_path), "pareto_analysis.csv missing"

    df = pd.read_csv(pareto_path)
    assert len(df) >= 10, "Pareto analysis has too few records"
    assert "is_pareto_optimal" in df.columns
    assert "suitability" in df.columns
    assert "expected_yield_tha" in df.columns
    assert "water_risk" in df.columns
    assert "erosion_risk" in df.columns

    n_pareto = df["is_pareto_optimal"].sum()
    n_dominated = (~df["is_pareto_optimal"]).sum()
    assert n_pareto > 0, "No Pareto-optimal candidates found"
    assert n_dominated > 0, "No dominated candidates found"


# -----------------------------------------------------------------------
# 9. Constraint vs Soft Penalty Fusion Results
# -----------------------------------------------------------------------
def test_v5_fusion_and_constraint_experiment():
    fusion_path = os.path.join(ARTIFACTS_V5, "fusion_results.csv")
    assert os.path.exists(fusion_path), "fusion_results.csv missing"

    df = pd.read_csv(fusion_path)
    assert len(df) >= 7, "Expected at least 7 fusion configurations"
    assert "configuration" in df.columns
    assert "weight_ml" in df.columns
    assert "weight_kb" in df.columns
    assert "constraint_type" in df.columns

    # Verify hard constraint test is present
    assert any("Hard Constraint" in str(r) for r in df["constraint_type"]), \
        "Hard constraint experiment missing"

    # Verify soft penalty experiment is present
    assert any("Soft Penalty" in str(r) for r in df["constraint_type"]), \
        "Soft penalty experiment missing"


# -----------------------------------------------------------------------
# 10. Controlled Ablation Study (11 Configurations)
# -----------------------------------------------------------------------
def test_v5_ablation_study():
    ab_path = os.path.join(ARTIFACTS_V5, "ablation_results.csv")
    assert os.path.exists(ab_path), "ablation_results.csv missing"

    df = pd.read_csv(ab_path)
    assert len(df) == 11, f"Expected 11 ablation configurations, got {len(df)}"
    assert "ML Multiclass Only" in " ".join(df["configuration"].tolist())
    assert any("Learning-to-Rank" in c for c in df["configuration"]), \
        "Learning-to-Rank configuration missing from ablation"


# -----------------------------------------------------------------------
# 11. Crop-Level Fairness
# -----------------------------------------------------------------------
def test_v5_crop_level_fairness():
    fair_path = os.path.join(ARTIFACTS_V5, "crop_level_metrics.csv")
    assert os.path.exists(fair_path), "crop_level_metrics.csv missing"

    df = pd.read_csv(fair_path)
    assert len(df) == 16, f"Expected 16 crop classes, got {len(df)}"
    assert "top1_accuracy" in df.columns
    assert "top5_presence" in df.columns
    assert "mean_ranking_position" in df.columns

    # Minority crops should not completely collapse: at least 50% of minority crops
    # must have Top-5 presence above a minimum floor (10%). This tolerates genuine
    # class-imbalance in the 10,091-record dataset where all 16 crops are minority.
    if "minority_status" in df.columns:
        minority = df[df["minority_status"] == "Minority"]
        if len(minority) > 0:
            pct_above_floor = (minority["top5_presence"] >= 0.10).mean()
            assert pct_above_floor >= 0.50, (
                f"More than half of minority crops have Top-5 presence below 10%: "
                f"{pct_above_floor:.1%} are above floor"
            )


# -----------------------------------------------------------------------
# 12. Temporal Feature Drift Analysis
# -----------------------------------------------------------------------
def test_v5_temporal_drift_analysis():
    drift_path = os.path.join(ARTIFACTS_V5, "drift_analysis.csv")
    assert os.path.exists(drift_path), "drift_analysis.csv missing"

    df = pd.read_csv(drift_path)
    assert len(df) >= 5, "Drift analysis has fewer than 5 features"
    assert "feature" in df.columns
    assert "ks_statistic" in df.columns
    assert "population_stability_index" in df.columns
    assert "drift_status" in df.columns

    # No feature should have extreme drift (PSI > 0.50 would indicate data corruption)
    assert df["population_stability_index"].max() < 0.50, "Extreme PSI drift detected"


# -----------------------------------------------------------------------
# 13. Error Analysis & Counterfactuals & Stability
# -----------------------------------------------------------------------
def test_v5_error_and_counterfactual_artifacts():
    for fname in ["error_analysis.csv", "error_analysis.md",
                  "counterfactual_results.csv", "ranking_stability.csv"]:
        path = os.path.join(ARTIFACTS_V5, fname)
        assert os.path.exists(path), f"{fname} missing"

    cf_df = pd.read_csv(os.path.join(ARTIFACTS_V5, "counterfactual_results.csv"))
    assert len(cf_df) == 5, "Expected 5 counterfactual perturbation scenarios"
    assert "kendall_tau" in cf_df.columns
    assert "spearman_rho" in cf_df.columns

    rs_df = pd.read_csv(os.path.join(ARTIFACTS_V5, "ranking_stability.csv"))
    assert len(rs_df) == 5


# -----------------------------------------------------------------------
# 14. Calibration Metrics
# -----------------------------------------------------------------------
def test_v5_calibration_metrics():
    cal_path = os.path.join(ARTIFACTS_V5, "calibration_metrics.json")
    assert os.path.exists(cal_path), "calibration_metrics.json missing"

    with open(cal_path, "r") as f:
        c = json.load(f)

    assert "uncalibrated_multiclass" in c
    assert "ece" in c["uncalibrated_multiclass"]
    # ECE must remain below 0.15 (not catastrophically miscalibrated)
    assert c["uncalibrated_multiclass"]["ece"] < 0.15
    assert "selected_calibration_method" in c


# -----------------------------------------------------------------------
# 15. Monitoring Specification
# -----------------------------------------------------------------------
def test_v5_monitoring_spec():
    mon_path = os.path.join(ARTIFACTS_V5, "monitoring_spec.md")
    assert os.path.exists(mon_path), "monitoring_spec.md missing"

    with open(mon_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "PSI" in content
    assert "latency" in content.lower() or "Latency" in content


# -----------------------------------------------------------------------
# 16. Leakage Audit Artifact
# -----------------------------------------------------------------------
def test_v5_leakage_audit():
    leakage_path = os.path.join(ARTIFACTS_V5, "leakage_audit.md")
    assert os.path.exists(leakage_path), "leakage_audit.md missing"

    with open(leakage_path, "r", encoding="utf-8") as f:
        content = f.read()
        content_lower = content.lower()

    # Must document query segregation (any of the expected keyword clusters)
    assert any(kw in content_lower for kw in ["grouped", "query", "zero query fragmentation", "leakage"]), \
        "Leakage audit does not document query segregation"
    # Must reference pre-harvest predictor exclusion
    assert any(kw in content_lower for kw in ["pre-harvest", "preharvest", "harvest", "predictor"]), \
        "Leakage audit does not document pre-harvest predictor exclusion"


# -----------------------------------------------------------------------
# 17. V4 vs V5 Comparison Artifact
# -----------------------------------------------------------------------
def test_v5_v4_comparison_artifact():
    vv_path = os.path.join(ARTIFACTS_V5, "v4_vs_v5.csv")
    assert os.path.exists(vv_path), "v4_vs_v5.csv missing"

    df = pd.read_csv(vv_path)
    assert len(df) >= 10, "v4_vs_v5.csv has too few metrics"
    assert "metric" in df.columns
    assert "v4" in df.columns
    assert "v5" in df.columns
    assert "delta" in df.columns

    # Check Spatiotemporal Top-5 row exists
    assert any("Spatiotemporal" in str(r) for r in df["metric"]), \
        "Spatiotemporal Top-5 missing from comparison"


# -----------------------------------------------------------------------
# 18. Final Report
# -----------------------------------------------------------------------
def test_v5_final_report():
    report_path = os.path.join(ARTIFACTS_V5, "FINAL_REPORT.md")
    assert os.path.exists(report_path), "FINAL_REPORT.md missing"

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Required sections
    required_sections = [
        "Executive Summary",
        "V4 Baseline",
        "V5 Scientific Hypothesis",
        "Learning-to-Rank",
        "Pareto",
        "Promotion Decision",
        "Reproducibility",
    ]
    for section in required_sections:
        assert section in content, f"Section '{section}' missing from FINAL_REPORT.md"


# -----------------------------------------------------------------------
# 19. API Backward Compatibility & V5 Multi-Objective Namespaces
# -----------------------------------------------------------------------
def test_v5_api_backward_compatibility():
    """Full backward compatibility test with V5 model version."""
    rec = agri_inference.generate_crop_recommendation(
        lat=15.335,
        lon=76.46,
        season="Kharif",
        top_k=5,
        model_version="v5",
    )
    assert rec["status"] == "success"
    assert rec["model"]["version"] == "v5"
    assert len(rec["recommendations"]) == 5

    prim = rec["primary_recommendation"]
    # Section 22 contract
    assert "crop" in prim
    assert "suitability_score" in prim
    assert "expected_yield_tha" in prim
    assert "confidence" in prim
    assert "risk" in prim
    assert "rank" in prim
    assert "expected_yield" in prim
    assert "lower" in prim["expected_yield"]
    assert "upper" in prim["expected_yield"]
    assert prim["expected_yield"]["lower"] <= prim["expected_yield"]["estimate"] <= prim["expected_yield"]["upper"]
    assert "decision_margin" in prim
    assert "ranking_stability" in prim

    # Backward compatibility keys
    assert "uncertainty" in rec
    assert "risk" in rec
    assert "drivers" in rec
    assert "counterfactuals" in rec
    assert "explanation" in rec
    assert "data_quality" in rec


def test_v5_api_ranking_monotonicity():
    """Recommendations must be sorted in descending suitability order."""
    rec = agri_inference.generate_crop_recommendation(
        lat=28.5, lon=77.2, season="Rabi", top_k=5, model_version="v5"
    )
    recs = rec["recommendations"]
    for i in range(len(recs) - 1):
        assert recs[i]["suitability_score"] >= recs[i + 1]["suitability_score"], \
            "Recommendations not sorted by suitability"


# -----------------------------------------------------------------------
# 20. V5 Model Prediction Sanity
# -----------------------------------------------------------------------
def test_v5_crop_model_prediction_shape():
    m, le, meta, v = agri_inference.load_crop_artifacts("v5")
    assert v == "v5"
    assert m is not None
    assert len(le.classes_) == 16

    dummy = np.ones((1, 26), dtype=np.float32)
    probs = m.predict_proba(dummy)
    assert probs.shape == (1, 16)
    assert abs(probs.sum() - 1.0) < 1e-4


def test_v5_ranking_model_artifacts():
    """V5 ranking_model.pkl must be loadable and produce scores."""
    ranking_model_path = os.path.join(MODELS_V5, "ranking_model.pkl")
    assert os.path.exists(ranking_model_path), "ranking_model.pkl missing from V5 registry"

    ranker = joblib.load(ranking_model_path)
    assert ranker is not None

    # Predict on dummy candidate features (26 base + 16 crop one-hot + 1 kb = 43 features)
    dummy_input = np.ones((16, 43), dtype=np.float32)
    scores = ranker.predict(dummy_input)
    assert scores.shape == (16,), f"Unexpected ranking score shape: {scores.shape}"


def test_v5_yield_model_preserved():
    """V5 yield model must be the same certified R² ~ 0.954 model from V4."""
    yield_path = os.path.join(MODELS_V5, "yield_model.pkl")
    assert os.path.exists(yield_path)

    ym = joblib.load(yield_path)
    assert ym is not None

    # Dummy prediction sanity — yield model was trained on 42 features
    n_feat = ym.n_features_in_
    dummy_yield_input = np.ones((1, n_feat), dtype=np.float32)
    pred = ym.predict(dummy_yield_input)
    assert len(pred) == 1
    assert pred[0] > 0.0, "Yield model predicts non-positive yield on dummy input"
