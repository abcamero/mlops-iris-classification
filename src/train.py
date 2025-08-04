# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_data(path="data/iris_processed.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_logistic_regression(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[LogisticRegression] Accuracy: {acc:.4f}, F1: {f1:.4f}")

def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="RandomForest"):
        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[RandomForest] Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)
