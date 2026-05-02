"""
utils.py - Backwards-compatibility shim.

The trained model (joblib) was serialised when HybridDeepModel lived in
this module. Python's pickle loader needs the original import path to
exist, so this file re-exports the class from its new location.

Once you retrain with ``python -m scripts.train``, this file can be removed.
"""

from src.model import HybridDeepModel  # noqa: F401
from src.data import symptoms_dict, diseases_list, label_encoder  # noqa: F401
