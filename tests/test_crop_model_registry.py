"""Tests for Crop Model Registry and Rollback Mechanisms."""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agri_inference


def test_registry_default_resolution():
    # Should resolve to v2 by default
    model_dir, version = agri_inference.get_model_dir()
    assert version == "v2"
    assert os.path.exists(model_dir)


def test_registry_explicit_v1_resolution():
    model_dir, version = agri_inference.get_model_dir("v1")
    assert version == "v1"
    assert os.path.exists(model_dir)
    assert "v1" in model_dir


def test_registry_unknown_fallback():
    # If a non-existent version is requested, must gracefully fall back to v1
    model_dir, version = agri_inference.get_model_dir("v99_non_existent")
    assert version == "v1"
    assert os.path.exists(model_dir)


def test_registry_caching():
    m1, le1, meta1, v1 = agri_inference.load_crop_artifacts("v2")
    m2, le2, meta2, v2 = agri_inference.load_crop_artifacts("v2")
    # Objects should be identical from cache
    assert m1 is m2
    assert le1 is le2
