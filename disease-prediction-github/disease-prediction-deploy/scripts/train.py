"""
scripts/train.py - Train the hybrid ensemble model and save to disk.

Usage
-----
    python -m scripts.train          # from project root
    python scripts/train.py          # alternative
"""

import os
import sys
import gc

import numpy as np
import pandas as pd
import joblib

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import DATA_PATH, label_encoder, symptoms_dict, diseases_list
from src.model import HybridDeepModel

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "ensemble_model_augmented.joblib")


def main():
    print("📦 Loading augmented dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("diseases", axis=1).values.astype(np.float32)
    y = label_encoder.transform(df["diseases"])

    del df
    gc.collect()

    model = HybridDeepModel(len(symptoms_dict), len(diseases_list))

    print("\n🧠 Training Hybrid Model (3 DL + 5 ML)...")
    model.fit(X, y, epochs=10, batch_size=32)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\n✅ Model trained and saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
