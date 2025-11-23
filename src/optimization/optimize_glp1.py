import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.generate_glp1_candidates import generate_single_mutants
from optimization.score_glp1_sequence import (
    score_sequence_for_diabetes,
    score_sequence_for_obesity,
    BASE_GLP1,
)


def optimize_for_diabetes(start_seq: str = BASE_GLP1, top_k: int = 5):
    """
    Generate single-mutation candidates around start_seq
    and return the top_k sequences ranked by Diabetes score.
    """
    candidates = generate_single_mutants(start_seq)
    results = []

    for cand in candidates:
        seq = cand["sequence"]
        score = score_sequence_for_diabetes(seq)
        results.append(
            {
                "sequence": seq,
                "position": cand["position"],
                "substitution": cand["substitution"],
                "score": score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def optimize_for_obesity(start_seq: str = BASE_GLP1, top_k: int = 5):
    """
    For now obesity uses the same scoring as diabetes.
    Later we can change the scoring weights.
    """
    candidates = generate_single_mutants(start_seq)
    results = []

    for cand in candidates:
        seq = cand["sequence"]
        score = score_sequence_for_obesity(seq)
        results.append(
            {
                "sequence": seq,
                "position": cand["position"],
                "substitution": cand["substitution"],
                "score": score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Simple manual test
    print("Top 5 diabetes-optimized single mutants from baseline GLP-1:")
    top_diab = optimize_for_diabetes()
    for i, cand in enumerate(top_diab, start=1):
        print(f"{i}. seq={cand['sequence']}  pos={cand['position']}  "
              f"sub={cand['substitution']}  score={cand['score']:.3f}")

    print("\nTop 5 obesity-optimized single mutants from baseline GLP-1:")
    top_ob = optimize_for_obesity()
    for i, cand in enumerate(top_ob, start=1):
        print(f"{i}. seq={cand['sequence']}  pos={cand['position']}  "
              f"sub={cand['substitution']}  score={cand['score']:.3f}")
