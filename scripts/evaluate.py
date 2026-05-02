"""
scripts/evaluate.py - Evaluate the trained ensemble model on a held-out test split.

Usage
-----
    python -m scripts.evaluate       # from project root
    python scripts/evaluate.py       # alternative
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import DATA_PATH, label_encoder, symptoms_dict, diseases_list

MODEL_FILE = os.path.join("models", "ensemble_model_augmented.joblib")


def main():
    print("📦 Loading trained model...")
    model = joblib.load(MODEL_FILE)

    print("📦 Loading dataset for evaluation...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("diseases", axis=1).values
    y = label_encoder.transform(df["diseases"])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Evaluating Hybrid Model on {len(X_test)} test samples...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Hybrid Ensemble Accuracy: {acc * 100:.2f}%")

    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(diseases_list.values())))


if __name__ == "__main__":
    main()
