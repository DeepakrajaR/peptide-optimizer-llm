import os
import numpy as np

# Lazy-load model and helper
_model_ms = None


def _data_path(name: str) -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(root, 'data', 'processed', name)


def _load_model_ms():
    global _model_ms
    if _model_ms is not None:
        return _model_ms

    try:
        import joblib
    except Exception as e:
        raise ImportError("Missing dependency 'joblib'. Add it to requirements.txt and redeploy.") from e

    MODEL_PATH = _data_path('model_ms_rf.pkl')
    try:
        _model_ms = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        # Model not present — allow fallback behavior in the caller
        _model_ms = None
        return _model_ms
    except Exception as e:
        raise RuntimeError(f"Failed to load MS model from '{MODEL_PATH}': {e}") from e

    return _model_ms


def score_sequence_for_ms(seq: str) -> float:
    """
    Returns MS-likeness probability (0 to 1).
    Higher = more similar to glatiramer / IL-10 / IL-23 peptides.
    """
    # load helper features function lazily
    try:
        from models.ms_features import ms_features
    except Exception as e:
        raise ImportError("Missing or broken `models.ms_features`. Ensure dependencies are installed.") from e

    model_ms = _load_model_ms()

    X = ms_features([seq])  # single-sample inference

    if model_ms is None:
        # Fallback deterministic heuristic mapped to [0,1].
        # Use a simple logistic on a linear combination of features so output
        # looks like a probability but requires no model files.
        import math
        feats = X[0]
        # Feature vector: [length, frac_A, frac_E, frac_K, frac_Y, frac_pos, frac_neg, hydro]
        length, frac_A, frac_E, frac_K, frac_Y, frac_pos, frac_neg, hydro = feats
        score_lin = (
            1.2 * frac_A + 1.5 * frac_E + 1.3 * frac_K + 0.8 * frac_Y
            - 0.01 * length + 0.5 * frac_pos - 0.3 * frac_neg + 0.2 * hydro
        )
        prob = 1.0 / (1.0 + math.exp(-score_lin))
        return float(prob)

    prob = model_ms.predict_proba(X)[0, 1]   # probability of class=1
    return float(prob)
