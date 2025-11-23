import os
import numpy as np

# --- 1. Define the baseline GLP-1 sequence ---
# Human GLP-1 (7-36)
BASE_GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"

# Module-level cache for lazy-loaded encoder & model
_encoder = None
_model = None


def _data_path(name: str) -> str:
    """Return absolute path to data/processed/<name> relative to repo root."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(root, 'data', 'processed', name)


def _load_encoder_and_model():
    """
    Lazy-load encoder and model. Raises ImportError with helpful message
    if required packages are missing.
    """
    global _encoder, _model
    if _encoder is not None and _model is not None:
        return _encoder, _model

    try:
        import joblib
    except Exception as e:
        raise ImportError(
            "Missing dependency 'joblib'. Add 'joblib' to requirements.txt and redeploy."
        ) from e

    try:
        # Import encoder implementation only when needed
        from models.features_glp1 import GLP1FeatureEncoder
    except Exception:
        # Allow joblib to load a persisted encoder even if source class is unavailable
        GLP1FeatureEncoder = None

    ENCODER_PATH = _data_path('glp1_encoder.pkl')
    MODEL_PATH = _data_path('model_glp1_diabetes_rf.pkl')

    try:
        _encoder = joblib.load(ENCODER_PATH)
        _model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        # Model files are not present — return None to allow a graceful fallback.
        _encoder, _model = None, None
        return _encoder, _model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load encoder/model from '{ENCODER_PATH}' and '{MODEL_PATH}': {e}"
        ) from e

    return _encoder, _model


# --- 3. Extract mutations from a user sequence ---
def extract_mutations(seq, base_seq=BASE_GLP1):
    """
    Compare user sequence to baseline GLP-1.
    Return list of (position, substitution) for each change.
    Positions returned are 1-based.
    """
    mutations = []
    min_len = min(len(seq), len(base_seq))

    for i in range(min_len):
        if seq[i] != base_seq[i]:
            pos = i + 1  # 1-based indexing
            sub = seq[i]  # new amino acid
            mutations.append((pos, sub))

    return mutations


# --- 4. Score sequence for diabetes ---
def score_sequence_for_diabetes(seq):
    mutations = extract_mutations(seq)

    if not mutations:
        return 0.0  # identical to baseline → neutral effect

    # Ensure encoder & model are available
    encoder, model = _load_encoder_and_model()

    if encoder is None or model is None:
        # Fallback heuristic when trained models are not available.
        # Deterministic simple scoring: give a modest boost for substitutions
        # to residues often considered favorable for peptide activity.
        HOT = set(list("AEKYFW"))
        base_len = len(BASE_GLP1)
        score = 0.0
        for pos, sub in mutations:
            pos_norm = (pos - 1) / max(1, base_len - 1)
            weight = 1.0 - pos_norm  # earlier positions slightly more important
            bonus = 1.0 if sub in HOT else 0.2
            score += weight * bonus * 0.1

        return float(score)

    # Build a temporary dataframe to feed the encoder
    import pandas as pd
    df = pd.DataFrame(mutations, columns=["Position", "Substitution"])

    X = encoder.transform(df)
    preds = model.predict(X)

    # Sum the effects from each mutation
    return float(np.sum(preds))


# --- 5. Score sequence for obesity ---
# For now: same as Diabetes (later we modify weighting)
def score_sequence_for_obesity(seq):
    return score_sequence_for_diabetes(seq)
