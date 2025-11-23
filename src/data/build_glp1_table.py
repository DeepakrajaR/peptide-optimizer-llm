import pandas as pd
import os

def main():
    # Paths
    raw_path = "data/raw/GLP1R_complete_approx.xlsx"
    out_path = "data/processed/glp1_substitutions_effects.csv"

    # Load Excel
    xls = pd.ExcelFile(raw_path)

    # Function to load a sheet into unified format
    def load_sheet(sheet_name):
        df = pd.read_excel(xls, sheet_name)

        # Sheets already follow: Position | Substitution | Endpoint | Approx_Value
        # So we simply return them
        return df[["Position", "Substitution", "Endpoint", "Approx_Value"]]

    # Load all substitution-effect sheets
    sheets = [
        "Fig3_DialIn_Approx",
        "Fig4_DMS_Approx",
        "Fig5_Glu_HLE_Approx",
        "Fig6_FineTune_Approx",
    ]

    dfs = [load_sheet(s) for s in sheets]

    # Combine into one dataframe
    df_all = pd.concat(dfs, ignore_index=True)

    # Pivot so each substitution has all endpoints
    df_pivot = df_all.pivot_table(
        index=["Position", "Substitution"],
        columns="Endpoint",
        values="Approx_Value",
        aggfunc="first",
    ).reset_index()

    # Ensure output folder exists
    os.makedirs("data/processed", exist_ok=True)

    # Save CSV
    df_pivot.to_csv(out_path, index=False)
    print(f"Saved processed GLP-1 substitution table → {out_path}")


if __name__ == "__main__":
    main()
