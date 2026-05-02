"""
model.py - Deep Learning architectures and HybridDeepModel ensemble class.
"""

import os
import gc

# Suppress TensorFlow info/warning logs (keep only errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data import diseases_list

# Limit TensorFlow GPU memory growth to avoid OOM
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)

# Directory where DL weight files are stored
WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "dl_weights"
)


# ======================================================================
# Deep Learning Architectures
# ======================================================================

def create_dense_model(input_dim: int, num_classes: int) -> keras.Model:
    """Fully-connected neural network (256 → 128 → softmax)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_conv_model(input_dim: int, num_classes: int) -> keras.Model:
    """1-D convolutional network for local symptom-cluster detection."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_attention_model(input_dim: int, num_classes: int) -> keras.Model:
    """Self-attention network that learns symptom importance weights."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    attention = layers.Dense(128, activation="softmax", name="attention_vec")(x)
    x = layers.Multiply()([x, attention])
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _clear_tf_memory():
    """Fully clear TensorFlow session memory between model trainings."""
    tf.keras.backend.clear_session()
    gc.collect()


# ======================================================================
# Hybrid Ensemble Model
# ======================================================================

class HybridDeepModel:
    """Ensemble of 3 DL + 5 ML classifiers with probability averaging.

    Training is memory-optimised: each DL model is trained sequentially,
    saved to disk, then freed before the next one begins.
    """

    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.num_classes = num_classes

        # DL models are created on-demand (not all at once) to save memory
        self.dl_models: dict = {}

        # Classical ML models
        self.ml_models = {
            "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
            "nb": GaussianNB(),
            "dt": DecisionTreeClassifier(),
            "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1),
            "xgb": XGBClassifier(
                use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1
            ),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y, epochs: int = 25, batch_size: int = 32):
        """Train all 8 sub-models (3 DL sequentially, then 5 ML on sampled data)."""
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        # ---- Dense NN ----
        print("\nTraining Dense Model...")
        _clear_tf_memory()
        dense = create_dense_model(self.input_dim, self.num_classes)
        dense.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        dense.save_weights(os.path.join(WEIGHTS_DIR, "dense.weights.h5"))
        del dense
        _clear_tf_memory()

        # ---- Conv1D ----
        print("\nTraining Conv1D Model...")
        conv = create_conv_model(self.input_dim, self.num_classes)
        X_conv = np.expand_dims(X, axis=-1)
        conv.fit(X_conv, y, epochs=epochs, batch_size=batch_size, verbose=1)
        conv.save_weights(os.path.join(WEIGHTS_DIR, "conv.weights.h5"))
        del conv, X_conv
        _clear_tf_memory()

        # ---- Attention ----
        print("\nTraining Attention Model...")
        attention = create_attention_model(self.input_dim, self.num_classes)
        attention.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
        attention.save_weights(os.path.join(WEIGHTS_DIR, "attention.weights.h5"))
        del attention
        _clear_tf_memory()

        # ---- ML classifiers (sampled: 10 rows per disease for speed) ----
        print("\nTraining ML Models (sampled data: 10 rows per disease)...")
        unique_classes = np.unique(y)
        sample_idx = []
        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            n = min(10, len(cls_idx))
            sample_idx.extend(np.random.choice(cls_idx, n, replace=False))

        X_sampled, y_sampled = X[sample_idx], y[sample_idx]
        print(f"  → Sampled {len(X_sampled)} rows from {len(unique_classes)} diseases")

        for name, ml_model in self.ml_models.items():
            print(f"  → {name.upper()}")
            ml_model.fit(X_sampled, y_sampled)

        # Reload DL models for ensemble prediction
        print("\nReloading DL models for ensemble prediction...")
        self._load_dl_models()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _load_dl_models(self):
        """Recreate DL architectures and load saved weights from disk."""
        _clear_tf_memory()
        self.dl_models = {
            "dense": create_dense_model(self.input_dim, self.num_classes),
            "conv": create_conv_model(self.input_dim, self.num_classes),
            "attention": create_attention_model(self.input_dim, self.num_classes),
        }
        self.dl_models["dense"].load_weights(os.path.join(WEIGHTS_DIR, "dense.weights.h5"))
        self.dl_models["conv"].load_weights(os.path.join(WEIGHTS_DIR, "conv.weights.h5"))
        self.dl_models["attention"].load_weights(os.path.join(WEIGHTS_DIR, "attention.weights.h5"))

    def predict(self, X):
        """Return predicted disease indices via 8-model probability averaging."""
        if not self.dl_models:
            self._load_dl_models()

        X_conv = np.expand_dims(X, axis=-1)
        preds = []

        # DL predictions
        preds.append(self.dl_models["dense"].predict(X, verbose=0))
        preds.append(self.dl_models["conv"].predict(X_conv, verbose=0))
        preds.append(self.dl_models["attention"].predict(X, verbose=0))

        # ML predictions
        for _, ml_model in self.ml_models.items():
            if hasattr(ml_model, "predict_proba"):
                preds.append(ml_model.predict_proba(X))
            else:
                onehot = np.zeros((len(X), len(diseases_list)))
                y_pred = ml_model.predict(X)
                onehot[np.arange(len(y_pred)), y_pred] = 1
                preds.append(onehot)

        avg_pred = np.mean(preds, axis=0)
        return np.argmax(avg_pred, axis=1)

    # ------------------------------------------------------------------
    # Serialisation (joblib/pickle)
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Exclude Keras models when pickling (weights live on disk)."""
        state = self.__dict__.copy()
        state["dl_models"] = {}
        return state

    def __setstate__(self, state):
        """DL models are reloaded lazily on first ``predict()`` call."""
        self.__dict__.update(state)
