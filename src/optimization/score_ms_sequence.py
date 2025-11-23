import joblib
import numpy as np
from models.ms_features import ms_features

MODEL_PATH = "data/processed/model_ms_rf.pkl"

# Load MS classifier once
model_ms = joblib.load(MODEL_PATH)


def score_sequence_for_ms(seq: str) -> float:
    """
    Returns MS-likeness probability (0 to 1).
    Higher = more similar to glatiramer / IL-10 / IL-23 peptides.
    """
    X = ms_features([seq])  # single-sample inference
    prob = model_ms.predict_proba(X)[0, 1]   # probability of class=1
    return float(prob)
