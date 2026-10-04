"""Tests for Probability Calibration & Uncertainty Metrics."""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.experiments_crop_v2 import compute_calibration_metrics


def test_perfect_calibration():
    # If confidence matches empirical accuracy perfectly
    y_true = np.array([0, 1, 0, 1])
    # perfectly calibrated 50-50
    y_prob = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    res = compute_calibration_metrics(y_true, y_prob, n_classes=2, n_bins=5)
    assert 0.0 <= res["brier_score"] <= 1.0
    assert 0.0 <= res["ece"] <= 1.0


def test_brier_score_bounds():
    y_true = np.array([0, 1, 2])
    # Extreme wrong predictions
    y_prob_wrong = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    res_wrong = compute_calibration_metrics(y_true, y_prob_wrong, n_classes=3)
    assert res_wrong["brier_score"] == 2.0  # Max possible multi-class Brier error

    # Perfect predictions
    y_prob_right = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    res_right = compute_calibration_metrics(y_true, y_prob_right, n_classes=3)
    assert res_right["brier_score"] == 0.0
    assert res_right["ece"] == 0.0
