# src/preprocess.py

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(save_path="data/iris_processed.csv"):
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled["target"] = y

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_scaled.to_csv(save_path, index=False)
    print(f"[INFO] Iris dataset preprocessed and saved to {save_path}")

if __name__ == "__main__":
    load_and_preprocess()
