# src/preprocess.py

import pandas as pd

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna()
    return df
