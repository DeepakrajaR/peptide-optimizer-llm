import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

from models.ms_features import ms_features


def main():
    in_path = "data/raw/ms_peptides.csv"
    out_model_path = "data/processed/model_ms_rf.pkl"

    # Load the MS peptide dataset
    df = pd.read_csv(in_path)

    # X = features from sequences
    X = ms_features(df["sequence"])
    y = df["label"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, out_model_path)
    print(f"Saved MS model → {out_model_path}")

    # Simple evaluation
    acc = clf.score(X_test, y_test)
    print(f"MS classifier accuracy on test set: {acc:.3f}")


if __name__ == "__main__":
    main()
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))