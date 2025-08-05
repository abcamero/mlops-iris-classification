# src/train.py

import pandas as pd
import numpy as np
import mlflow
import os
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from preprocess import load_and_preprocess_data


mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Iris-Experiment-Final")

def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")  # artifact_path = "model"
        
        print(f"✔️ Logged {model_name} | F1: {f1:.4f}")

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "iris_processed.csv")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_log(LogisticRegression(max_iter=200), "LogisticRegression", X_train, X_test, y_train, y_test)
    train_and_log(RandomForestClassifier(), "RandomForest", X_train, X_test, y_train, y_test)
