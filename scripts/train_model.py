"""
train_model.py - Train and save the invoice classification model.
Run this once before starting the API: python scripts/train_model.py
"""

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# ─── Text Preprocessing ───────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase, remove special chars, strip extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Load Data ─────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "model.pkl")

df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].apply(preprocess)

X = df["text"].tolist()
y = df["category"].tolist()

print(f"✅ Loaded {len(X)} samples across {df['category'].nunique()} categories")
print(f"   Categories: {sorted(df['category'].unique())}\n")

# ─── Train / Test Split ───────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Build Pipeline ───────────────────────────────────────────────────────────

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,
        sublinear_tf=True,    # log-scaled TF
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",

    )),
])

# ─── Train ────────────────────────────────────────────────────────────────────

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("─── Evaluation ───────────────────────────────────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, preds):.2%}\n")
print(classification_report(y_test, preds))

# ─── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model saved → {MODEL_PATH}")
