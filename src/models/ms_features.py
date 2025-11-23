import numpy as np
import pandas as pd

# Simple Kyte-Doolittle hydrophobicity scale
HYDRO = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

def _aa_fraction(seq: str, aa: str) -> float:
    if not seq:
        return 0.0
    return seq.count(aa) / len(seq)

def _avg_hydrophobicity(seq: str) -> float:
    if not seq:
        return 0.0
    vals = [HYDRO.get(a, 0.0) for a in seq]
    return float(np.mean(vals))

def _charge_proxy(seq: str):
    """
    Very rough charge proxies:
    - Positive AA: K, R, H
    - Negative AA: D, E
    Returns (frac_positive, frac_negative)
    """
    if not seq:
        return 0.0, 0.0
    pos = sum(seq.count(a) for a in ["K", "R", "H"])
    neg = sum(seq.count(a) for a in ["D", "E"])
    L = len(seq)
    return pos / L, neg / L

def ms_features(seqs):
    """
    seqs: iterable/list/Series of peptide sequences (str)
    Returns: numpy array [n_samples, n_features]
    """
    rows = []

    for seq in seqs:
        seq = str(seq).strip().upper()

        length = len(seq)
        frac_A = _aa_fraction(seq, "A")
        frac_E = _aa_fraction(seq, "E")
        frac_K = _aa_fraction(seq, "K")
        frac_Y = _aa_fraction(seq, "Y")

        frac_pos, frac_neg = _charge_proxy(seq)
        hydro = _avg_hydrophobicity(seq)

        rows.append(
            [
                length,
                frac_A,
                frac_E,
                frac_K,
                frac_Y,
                frac_pos,
                frac_neg,
                hydro,
            ]
        )

    return np.array(rows)
