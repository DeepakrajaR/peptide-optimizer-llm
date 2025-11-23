import pandas as pd
import os

def main():
    in_path = "data/processed/glp1_substitutions_effects.csv"
    out_path = "data/processed/glp1_substitutions_labeled.csv"

    # Load the pivoted substitutions table
    df = pd.read_csv(in_path)

    # Some rows have EC50-style endpoints, some have pEC50-style.
    # We'll create combined "effect" columns by preferring EC50-style,
    # then falling back to pEC50-style if EC50 is missing.

    # If these columns don't exist for some rows, fillna will handle that.
    glp1r_ec50   = df.get("hGLP1R_EC50")
    glp1r_pec50  = df.get("hGLP1R_pEC50")
    sctr_ec50    = df.get("hSCTR_EC50")
    sctr_pec50   = df.get("hSCTR_pEC50")

    # Combined effect columns
    df["GLP1R_effect"] = glp1r_ec50.fillna(glp1r_pec50)
    df["SCTR_effect"]  = sctr_ec50.fillna(sctr_pec50)

    # Define targets:
    # - Higher GLP1R_effect is beneficial (better potency)
    # - Lower SCTR_effect is beneficial (less off-target), so we take negative
    df["GLP1R_benefit"] = df["GLP1R_effect"]
    df["SCTR_penalty"]  = -df["SCTR_effect"]

    # Drop rows that have no GLP1R info at all
    df = df.dropna(subset=["GLP1R_benefit"])

    # Make sure output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved labeled GLP-1 substitutions → {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
