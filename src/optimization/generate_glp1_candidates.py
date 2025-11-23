import pandas as pd
from optimization.score_glp1_sequence import BASE_GLP1

SUB_TABLE_PATH = "data/processed/glp1_substitutions_labeled.csv"

# Load substitution data once at import time
df_subs = pd.read_csv(SUB_TABLE_PATH)

# Build: position -> list of allowed substitutions (strings)
allowed_by_pos = (
    df_subs.groupby("Position")["Substitution"]
           .apply(lambda s: sorted(set(s)))
           .to_dict()
)

def generate_single_mutants(start_seq: str = BASE_GLP1):
    """
    Generate single-point mutants of the starting GLP-1 sequence
    using only substitutions that appear in the GLP-1 dataset.

    Returns a list of dicts:
    [
      {"sequence": "...", "position": 12, "substitution": "Y"},
      ...
    ]
    """
    candidates = []

    for pos, subs in allowed_by_pos.items():
        idx = pos - 1  # convert 1-based to 0-based index

        # Skip if position is outside the sequence
        if idx < 0 or idx >= len(start_seq):
            continue

        for sub in subs:
            # For now, only allow simple one-letter substitutions
            # (we'll ignore things like "Y+HLE" for this first version)
            if len(sub) != 1:
                continue

            # Skip if this mutation would not change the residue
            if start_seq[idx] == sub:
                continue

            new_seq_list = list(start_seq)
            new_seq_list[idx] = sub
            new_seq = "".join(new_seq_list)

            candidates.append(
                {
                    "sequence": new_seq,
                    "position": pos,
                    "substitution": sub,
                }
            )

    return candidates
