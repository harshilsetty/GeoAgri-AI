"""Erosion Model Regression Test Suite (Phase 1 & Phase 15).

Verifies that the existing production erosion model:
1. Retains identical feature schema and ordering.
2. Accurately reproduces the verified 90.50% holdout accuracy baseline on terrain_model/erosion_dataset.csv.
3. Produces finite probability outputs within [0.0, 1.0].
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "terrain_model", "erosion_model.pkl")
DATASET_PATH = os.path.join(ROOT, "terrain_model", "erosion_dataset.csv")

EXPECTED_FEATURES = [
    "slope",
    "vegetation",
    "elevation",
    "rainfall",
    "soil",
    "boulders",
    "ruins",
    "structures",
]


def test_erosion_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Erosion model not found at {MODEL_PATH}"
    assert os.path.exists(DATASET_PATH), f"Erosion dataset not found at {DATASET_PATH}"


def test_erosion_model_feature_schema():
    model = joblib.load(MODEL_PATH)
    assert getattr(model, "n_features_in_", 8) == 8, "Erosion model must expect 8 features"
    feature_names = list(getattr(model, "feature_names_in_", []))
    assert feature_names == EXPECTED_FEATURES, f"Expected {EXPECTED_FEATURES}, got {feature_names}"


def test_erosion_model_baseline_reproduction():
    df = pd.read_csv(DATASET_PATH)
    X = df[EXPECTED_FEATURES]
    y = df["erosion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds))
    rec = float(recall_score(y_test, preds))
    f1 = float(f1_score(y_test, preds))

    # Verify exact reproduction of benchmark metrics
    assert round(acc, 3) == 0.905, f"Expected accuracy 0.905, got {acc}"
    assert round(prec, 3) == 0.909, f"Expected precision 0.909, got {prec}"
    assert round(rec, 3) == 0.900, f"Expected recall 0.900, got {rec}"
    assert round(f1, 3) == 0.905, f"Expected F1 0.905, got {f1}"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be bounded in [0, 1]"
