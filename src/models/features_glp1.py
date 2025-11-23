import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class GLP1FeatureEncoder:
    def __init__(self):
        # Support both older and newer scikit-learn versions:
        # - older versions accept `sparse=False`
        # - scikit-learn >=1.2 removed `sparse` in favor of `sparse_output`
        try:
            self.enc_sub = OneHotEncoder(sparse=False, handle_unknown="ignore")
        except TypeError:
            self.enc_sub = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        
        # Positions vary from ~2–27; encode as numeric scaled feature
        self.max_position = 30.0

    def fit(self, df):
        # Fit only on the Substitution column
        self.enc_sub.fit(df[["Substitution"]])
        return self

    def transform(self, df):
        # One-hot encode substitutions
        sub_enc = self.enc_sub.transform(df[["Substitution"]])

        # Normalize positions
        pos_norm = (df["Position"].values.reshape(-1, 1)) / self.max_position

        # Combine features
        return np.hstack([pos_norm, sub_enc])

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
