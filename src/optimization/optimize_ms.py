import random
from optimization.score_ms_sequence import score_sequence_for_ms

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def generate_ms_single_mutants(start_seq: str):
    """
    Generate simple single-point mutants for MS optimization.
    Try all 20 amino acids at each position.
    """
    candidates = []
    seq_list = list(start_seq)

    for i in range(len(seq_list)):
        for aa in AMINO_ACIDS:
            if seq_list[i] == aa:
                continue
            new_seq = seq_list.copy()
            new_seq[i] = aa
            candidates.append("".join(new_seq))

    return candidates


def optimize_for_ms(start_seq: str, top_k: int = 5):
    """
    Generate MS-optimized sequences using MS-likeness score.
    Returns top_k sequences with highest MS probability.
    """
    candidates = generate_ms_single_mutants(start_seq)

    results = []
    for seq in candidates:
        score = score_sequence_for_ms(seq)
        results.append({"sequence": seq, "score": score})

    # Sort by descending MS score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    # Simple test
    print("Top 5 MS-optimized mutants for a test peptide:")
    start = "AEKAEKAEKAEKAAAKAEK"  # glatiramer-ish
    top = optimize_for_ms(start)
    for i, item in enumerate(top, start=1):
        print(f"{i}. seq={item['sequence']}  score={item['score']:.3f}")
