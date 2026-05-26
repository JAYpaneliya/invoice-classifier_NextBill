"""
app/main.py - Invoice Expense Classifier API
"""

import pickle
import os
import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"

# ─── Load Model ───────────────────────────────────────────────────────────────

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found. Please run: python scripts/train_model.py"
    )

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Invoice Expense Classifier",
    description="Classifies invoice text into expense categories using TF-IDF + Logistic Regression.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., example="Blue Dart courier charges for warehouse delivery")

class CategoryScore(BaseModel):
    category: str
    confidence: float

class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: List[CategoryScore]

class BatchRequest(BaseModel):
    texts: List[str]

class BatchResponse(BaseModel):
    results: List[PredictResponse]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Invoice Classifier API is running."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/categories", tags=["Info"])
def get_categories():
    """Return the list of supported expense categories."""
    return {"categories": pipeline.classes_.tolist()}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Classify a single invoice text into an expense category.
    Returns the predicted category, confidence score, and all class probabilities.
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")

    clean = preprocess(request.text)
    proba = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_

    top_idx = proba.argmax()
    all_scores = sorted(
        [CategoryScore(category=c, confidence=round(float(p), 4)) for c, p in zip(classes, proba)],
        key=lambda x: x.confidence,
        reverse=True,
    )

    return PredictResponse(
        category=classes[top_idx],
        confidence=round(float(proba[top_idx]), 4),
        all_scores=all_scores,
    )

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """Classify multiple invoice texts in one request."""
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts list cannot be empty.")

    results = []
    for text in request.texts:
        clean = preprocess(text)
        proba = pipeline.predict_proba([clean])[0]
        classes = pipeline.classes_
        top_idx = proba.argmax()
        all_scores = sorted(
            [CategoryScore(category=c, confidence=round(float(p), 4)) for c, p in zip(classes, proba)],
            key=lambda x: x.confidence,
            reverse=True,
        )
        results.append(PredictResponse(
            category=classes[top_idx],
            confidence=round(float(proba[top_idx]), 4),
            all_scores=all_scores,
        ))

    return BatchResponse(results=results)
