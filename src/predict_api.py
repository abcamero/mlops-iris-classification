# src/predict_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()

# MLflow registry details
MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "iris_classifier"
MODEL_STAGE = "Production"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from the registry
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

# Input format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "🚀 Iris Classifier FastAPI is running!"}

@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        input_df = pd.DataFrame([{
            "sepal length (cm)": input_data.sepal_length,
            "sepal width (cm)": input_data.sepal_width,
            "petal length (cm)": input_data.petal_length,
            "petal width (cm)": input_data.petal_width
        }])
        prediction = model.predict(input_df)
        return {"prediction": str(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

