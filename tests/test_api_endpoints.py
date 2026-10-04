"""FastAPI Endpoint Integration Tests (Phase 15).

Verifies /api/agriculture/recommend (GET & POST) and /api/predict?action=crop.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend_api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_agriculture_recommend_get():
    response = client.get("/api/agriculture/recommend?lat=15.335&lon=76.46&season=Kharif&topK=5")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
    assert "primary_recommendation" in data
    assert "data_quality" in data


def test_agriculture_recommend_post():
    payload = {
        "latitude": 22.57,
        "longitude": 88.36,
        "season": "Kharif",
        "topK": 5,
    }
    response = client.post("/api/agriculture/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["location"]["season"] == "Kharif"


def test_api_predict_crop_action():
    payload = {
        "latitude": 15.335,
        "longitude": 76.46,
        "season": "Kharif",
    }
    response = client.post("/api/predict?action=crop", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "primary_recommendation" in data
    assert "erosion" in data
