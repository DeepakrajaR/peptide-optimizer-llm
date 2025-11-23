import sys
import os
sys.path.append(os.path.abspath("src"))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from models.features_glp1 import GLP1FeatureEncoder

def main():
    # Load labeled substitution table
    df = pd.read_csv("data/processed/glp1_substitutions_labeled.csv")

    # Define X and y for the "Diabetes potency" model
    # Benefit target = GLP1R_benefit
    y = df["GLP1R_benefit"].values

    # Create & fit the feature encoder
    encoder = GLP1FeatureEncoder()
    X = encoder.fit_transform(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Save encoder + model
    joblib.dump(encoder, "data/processed/glp1_encoder.pkl")
    joblib.dump(model, "data/processed/model_glp1_diabetes_rf.pkl")

    # Print simple evaluation
    score = model.score(X_test, y_test)
    print(f"Diabetes GLP-1 model R^2 on test set: {score:.3f}")

if __name__ == "__main__":
    main()
