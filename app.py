import os
import sys

# Add src/ to the Python path so we can import optimization, models, etc.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from optimization.optimize_glp1 import optimize_for_diabetes, optimize_for_obesity
from optimization.optimize_ms import optimize_for_ms

import streamlit as st
import requests
import socket

API_URL = "http://127.0.0.1:8000"  # FastAPI backend


def _call_backend(payload: dict, timeout: float = 2.0):
    """
    Try to call the FastAPI backend. If it's unavailable, return None.
    """
    try:
        resp = requests.post(f"{API_URL}/optimize", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (requests.RequestException, socket.timeout):
        return None

st.set_page_config(
    page_title="Peptide Optimization Chatbot",
    page_icon="🧬",
    layout="centered",
)

st.title("🧬 Peptide Optimization Chatbot")

st.markdown(
    """
This tool suggests **optimized peptide sequences** for:

- **Diabetes** (GLP-1 based)
- **Obesity** (GLP-1 based)
- **Multiple Sclerosis** (glatiramer / IL-10 / IL-23 based)
"""
)

# --- Input section ---
disease = st.selectbox(
    "Select indication:",
    ["Diabetes", "Obesity", "Multiple Sclerosis"],
)

default_seq = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR" if disease in ["Diabetes", "Obesity"] else "AEKAEKAEKAEKAAAKAEK"

sequence = st.text_area(
    "Starting peptide sequence (one-letter amino acid code):",
    value=default_seq,
    height=100,
)

top_k = st.slider("Number of optimized candidates to show:", 1, 10, 5)

if st.button("Optimize"):
    if not sequence.strip():
        st.error("Please enter a peptide sequence.")
    else:
        payload = {
            "disease": disease.lower() if disease != "Multiple Sclerosis" else "ms",
            "starting_sequence": sequence.strip(),
            "top_k": top_k,
        }
        # First try the remote backend (useful during local development when FastAPI is running)
        data = _call_backend(payload)

        # Fallback to local optimization functions when backend is not reachable
        if data is None:
            st.info("Backend not reachable — running local optimization (this may load model files).")
            disease_key = payload["disease"]
            if disease_key == "diabetes":
                candidates = optimize_for_diabetes(payload["starting_sequence"], payload["top_k"])
            elif disease_key == "obesity":
                candidates = optimize_for_obesity(payload["starting_sequence"], payload["top_k"])
            else:
                candidates = optimize_for_ms(payload["starting_sequence"], payload["top_k"])

            data = {"candidates": candidates}

        st.subheader("Optimized candidates")
        candidates = data.get("candidates", [])

        if not candidates:
            st.warning("No candidates returned.")
        else:
            for i, cand in enumerate(candidates, start=1):
                st.markdown(f"### Candidate {i}")
                st.code(cand.get("sequence", ""), language="text")
                score = cand.get("score", None)
                if score is not None:
                    st.write(f"**Score:** {score:.3f}")
                pos = cand.get("position", None)
                sub = cand.get("substitution", None)
                if pos is not None and sub is not None:
                    st.write(f"Mutation: position {pos} → {sub}")

                # Simple explanation block
                with st.expander("Explain this candidate"):
                    if payload["disease"] in ["diabetes", "obesity"]:
                        st.write(
                            f"This candidate differs from your starting sequence at "
                            f"position {pos} with substitution **{sub}**.\n\n"
                            f"The model predicts this change increases the GLP-1 receptor benefit "
                            f"score for {payload['disease']} compared to the baseline."
                        )
                    else:
                        st.write(
                            "This candidate is predicted to be more 'MS-like' based on its "
                            "amino acid composition (especially A/E/K/Y content) and charge and "
                            "hydrophobicity patterns, making it more similar to known "
                            "glatiramer / IL-10 / IL-23 peptides."
                        )

st.markdown("---")
st.caption("Backend: FastAPI + ML models trained on GLP-1, glatiramer, and IL-10/IL-23-like peptides.")
