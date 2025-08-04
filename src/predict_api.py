# src/predict_api.py

import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load the Production model from MLflow Model Registry
model_uri = "models:/iris_classifier/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Initialize FastAPI app
app = FastAPI(title="Iris Classifier API")

# Define input schema using Pydantic
class IrisInput(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(input_data: IrisInput):
    # Convert list to DataFrame with one row
    input_df = [input_data.features]
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
