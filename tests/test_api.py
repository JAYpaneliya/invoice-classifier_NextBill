"""
tests/test_api.py - Unit tests for the Invoice Classifier API
Run with: pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

# ─── Ensure model exists before importing app ─────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "model.pkl")
if not os.path.exists(MODEL_PATH):
    pytest.skip("Model not trained yet. Run: python scripts/train_model.py", allow_module_level=True)

from app.main import app

client = TestClient(app)

# ─── Health ───────────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_categories():
    r = client.get("/categories")
    assert r.status_code == 200
    cats = r.json()["categories"]
    expected = {"Logistics", "Office Supplies", "Cloud/Software", "Utilities", "Travel", "Inventory"}
    assert expected.issubset(set(cats))

# ─── Predict ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_category", [
    ("Blue Dart courier charges for warehouse delivery", "Logistics"),
    ("AWS monthly cloud hosting bill", "Cloud/Software"),
    ("A4 paper reams purchase for office", "Office Supplies"),
    ("Flight tickets Mumbai to Delhi team offsite", "Travel"),
    ("Electricity bill for office premises", "Utilities"),
    ("Raw material purchase steel sheets", "Inventory"),
])
def test_predict_known_categories(text, expected_category):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == expected_category
    assert 0.0 <= body["confidence"] <= 1.0
    # all_scores should sum to ~1
    total = sum(s["confidence"] for s in body["all_scores"])
    assert abs(total - 1.0) < 0.01

def test_predict_returns_all_scores():
    r = client.post("/predict", json={"text": "Google Workspace license renewal"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["all_scores"]) == 6  # 6 categories
    # Sorted descending
    scores = [s["confidence"] for s in body["all_scores"]]
    assert scores == sorted(scores, reverse=True)

def test_predict_empty_text():
    r = client.post("/predict", json={"text": "  "})
    assert r.status_code == 422

def test_predict_missing_field():
    r = client.post("/predict", json={})
    assert r.status_code == 422

# ─── Batch Predict ────────────────────────────────────────────────────────────

def test_batch_predict():
    texts = [
        "DHL shipment fees for international cargo",
        "Microsoft 365 subscription for 50 users",
        "Hotel accommodation for sales conference",
    ]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert results[0]["category"] == "Logistics"
    assert results[1]["category"] == "Cloud/Software"
    assert results[2]["category"] == "Travel"

def test_batch_predict_empty():
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422
