from fastapi import FastAPI
from pydantic import BaseModel

# Import optimization engines (package-relative)
from ..optimization.optimize_glp1 import optimize_for_diabetes, optimize_for_obesity
from ..optimization.optimize_ms import optimize_for_ms

app = FastAPI(
    title="Peptide Optimization API",
    description="API for optimized peptide design for Diabetes, Obesity, and Multiple Sclerosis",
    version="1.0.0"
)


# ---------- REQUEST MODELS ----------

class OptimizeRequest(BaseModel):
    disease: str               # "diabetes" | "obesity" | "ms"
    starting_sequence: str     # peptide sequence (one-letter code)
    top_k: int = 5             # number of candidates to return


# ---------- ROUTES ----------

@app.get("/")
def root():
    return {"message": "Peptide Optimization API is running."}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    disease = req.disease.lower()

    if disease == "diabetes":
        result = optimize_for_diabetes(req.starting_sequence, req.top_k)

    elif disease == "obesity":
        result = optimize_for_obesity(req.starting_sequence, req.top_k)

    elif disease == "ms":
        result = optimize_for_ms(req.starting_sequence, req.top_k)

    else:
        return {"error": f"Unknown disease type: {req.disease}"}

    return {
        "disease": disease,
        "starting_sequence": req.starting_sequence,
        "top_k": req.top_k,
        "candidates": result
    }
