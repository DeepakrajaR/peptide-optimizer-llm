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
    prob = model_ms.predict_proba(X)[0, 1]   # probability of class=1
    return float(prob)
