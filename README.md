
# Peptide Optimizer LLM

An AI-powered peptide optimization platform for Diabetes, Obesity, and Multiple Sclerosis.  
Uses machine learning models trained on GLP‑1 substitution datasets, glatiramer-like peptides,  
and IL‑10 / IL‑23 immunomodulatory sequences.  
Includes a Streamlit-based interactive UI for exploring optimized peptide variants.

---

## 🚀 Features

### 🔹 Disease-Specific Optimization
- **Diabetes:** GLP‑1 receptor potency modeling using real substitution effect data  
- **Obesity:** GLP‑1 optimization with modifiable scoring weights  
- **Multiple Sclerosis:** MS-likeness classifier using glatiramer & IL‑10/23 inspired sequences  

### 🔹 ML Components
- Random Forest regression for GLP‑1 potency  
- Random Forest classifier for MS immunomodulatory similarity  
- Custom feature engineering for peptide sequences  

### 🔹 Optimization Engine
- Generates mutation candidates  
- Scores & ranks peptide variants  
- Provides interpretable reasoning for each optimized sequence  

### 🔹 Web Application (Streamlit)
- User selects indication  
- Inputs peptide sequence  
- Receives top optimized variants with explanations  

---

## 🧬 Project Structure

```
peptide-optimizer-llm/
│
├── app.py                   # Streamlit UI
├── requirements.txt         # Dependencies
├── data/
│   ├── raw/                 # Original datasets (GLP‑1 Excel, MS peptides)
│   └── processed/           # Saved ML models & engineered data
│
├── src/
│   ├── models/              # Training scripts & feature engineering
│   ├── optimization/        # Optimization engines for GLP‑1 & MS
│   └── app/                 # (Optional) FastAPI backend
│
└── README.md
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

(Optional) activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

---

## ▶️ Running the Streamlit App

```bash
streamlit run app.py
```

---

## 📁 Data Requirements

Place these files inside `data/raw/`:

- `GLP1R_complete_approx.xlsx`  
- `ms_peptides.csv`

The trained models will appear in:

```
data/processed/
  glp1_encoder.pkl
  model_glp1_diabetes_rf.pkl
  model_ms_rf.pkl
```

---

## 🧪 Training (Optional)

Run:

```bash
python src/models/train_glp1_models.py
python src/models/train_ms_model.py
```

---

## 🌐 Deployment

This project can be deployed for free using **Hugging Face Spaces**:

- Select **Streamlit** as the runtime  
- Upload `app.py`, `requirements.txt`, `src/`, and `data/processed/`  
- The UI becomes available instantly  

---

## © License

MIT License.  
Use for research, biotechnology prototyping, and educational purposes.

---

## 👨‍💻 Author

Project generated with guidance from ChatGPT.  
