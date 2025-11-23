import joblib
import numpy as np
from models.features_glp1 import GLP1FeatureEncoder

# --- 1. Define the baseline GLP-1 sequence ---
# Human GLP-1 (7-36)
BASE_GLP1 = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"

# --- 2. Load encoder & model only once ---
ENCODER_PATH = "data/processed/glp1_encoder.pkl"
MODEL_PATH   = "data/processed/model_glp1_diabetes_rf.pkl"

encoder = joblib.load(ENCODER_PATH)
model   = joblib.load(MODEL_PATH)


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
