"""Tests for Crop Intelligence V3 Ranking Engine (Phase D, G, P)."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train_crop_v3 import compute_ranking_metrics_v3


def test_ranking_metrics_calculation():
    # Synthetic test with 3 samples and 4 classes
    y_true = np.array([0, 1, 2])
    # Scores where sample 0 has rank 1 (prob 0.8 at idx 0)
    # Sample 1 has rank 2 (prob 0.7 at idx 0, prob 0.6 at idx 1)
    # Sample 2 has rank 3 (idx 0=0.5, idx 1=0.4, idx 2=0.3)
    scores = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.7, 0.6, 0.1, 0.0],
        [0.5, 0.4, 0.3, 0.1],
    ])
    labels = [0, 1, 2, 3]

    metrics = compute_ranking_metrics_v3(y_true, scores, labels)
    assert metrics["top_1"] == round(1.0 / 3.0, 4)
    assert metrics["top_3"] == 1.0  # all 3 are within top-3
    assert metrics["top_5"] == 1.0
    assert metrics["mrr"] > 0.0
    assert metrics["ndcg_3"] > 0.0
    assert metrics["ndcg_5"] > 0.0


def test_v3_ranking_artifacts_exist():
    ranking_csv = os.path.join(ROOT, "artifacts", "crop_v3", "ranking_metrics.csv")
    v2_v3_csv = os.path.join(ROOT, "artifacts", "crop_v3", "v2_vs_v3.csv")
    assert os.path.exists(ranking_csv), "ranking_metrics.csv must exist"
    assert os.path.exists(v2_v3_csv), "v2_vs_v3.csv must exist"

    df_rank = pd.read_csv(ranking_csv)
    assert len(df_rank) >= 5, "Must benchmark at least Systems A through E"
    systems = df_rank["system"].tolist()
    assert any("System_A" in s for s in systems)
    assert any("System_B" in s for s in systems)
    assert any("System_C" in s for s in systems)
    assert any("System_D" in s for s in systems)
    assert any("System_E" in s for s in systems)


def test_v3_recommendations_sorted_descending():
    import agri_inference
    rec = agri_inference.generate_crop_recommendation(15.335, 76.46, season="Kharif", top_k=5)
    assert "recommendations" in rec
    recs = rec["recommendations"]
    assert len(recs) == 5

    # Verify scores are monotonically non-increasing
    scores = [r["suitability_score"] for r in recs]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Recommendations not sorted: {scores}"
