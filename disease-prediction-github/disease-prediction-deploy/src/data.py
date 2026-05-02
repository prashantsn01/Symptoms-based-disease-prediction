"""
data.py - Dataset loading, label encoding, and symptom/disease mappings.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Path to the dataset (relative to project root)
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Final_Augmented_dataset_Diseases_and_Symptoms.csv",
)


def get_mappings():
    """Load the dataset and return symptom/disease dictionaries and encoder.

    Returns
    -------
    symptoms_dict : dict
        Mapping of symptom name (str) → column index (int).
    diseases_list : dict
        Mapping of encoded label (int) → disease name (str).
    encoder : LabelEncoder
        Fitted sklearn LabelEncoder for the ``diseases`` column.
    """
    df = pd.read_csv(DATA_PATH)
    encoder = LabelEncoder()
    df["prognosis_encoded"] = encoder.fit_transform(df["diseases"])

    diseases_list = {i: name for i, name in enumerate(encoder.classes_)}
    symptoms = df.columns.drop(["diseases", "prognosis_encoded"])
    symptoms_dict = {symptom: i for i, symptom in enumerate(symptoms)}

    return symptoms_dict, diseases_list, encoder


# ---------------------------------------------------------------------------
# Module-level singletons – imported everywhere else via:
#   from src.data import symptoms_dict, diseases_list, label_encoder
# ---------------------------------------------------------------------------
try:
    symptoms_dict, diseases_list, label_encoder = get_mappings()
except Exception as exc:
    print(f"⚠️ Dataset not loaded (web mode): {exc}")
    symptoms_dict: dict = {}
    diseases_list: dict = {}
    label_encoder = None
