"""Tests for Top-K, MRR, and NDCG Ranking Metrics."""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.experiments_crop_v2 import compute_ranking_metrics


def test_ranking_metrics_ideal():
    labels = [0, 1, 2, 3, 4]
    y_true = np.array([0, 1, 2])
    # Perfect top-1 predictions
    y_prob = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.05, 0.05, 0.8, 0.05, 0.05],
    ])
    res = compute_ranking_metrics(y_true, y_prob, labels)
    assert res["top_1"] == 1.0
    assert res["top_3"] == 1.0
    assert res["top_5"] == 1.0
    assert res["mrr"] == 1.0
    assert res["ndcg_3"] == 1.0
    assert res["ndcg_5"] == 1.0


def test_ranking_metrics_rank2_behavior():
    labels = [0, 1, 2, 3, 4]
    # True label is ranked 2nd
    y_true = np.array([0])
    y_prob = np.array([[0.3, 0.6, 0.05, 0.03, 0.02]])  # True class 0 is 2nd

    res = compute_ranking_metrics(y_true, y_prob, labels)
    assert res["top_1"] == 0.0
    assert res["top_3"] == 1.0
    assert res["top_5"] == 1.0
    assert res["mrr"] == 0.5  # 1/2
    assert res["ndcg_3"] == round(1.0 / np.log2(2 + 1), 4)
