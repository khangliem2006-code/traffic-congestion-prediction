from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd

app = FastAPI(
    title="Traffic Congestion Prediction API",
    description="""
This REST API predicts urban traffic congestion levels using
the XGBoost machine learning model trained on the
NYC Yellow Taxi Trip Dataset.

The API accepts taxi trip information as input
and returns the predicted congestion level.
""",
    version="1.0",
    contact={
        "name": "Traffic Congestion Prediction Project"
    }
)


model = joblib.load(
    "results/xgboost_model.pkl"
)

@app.get("/")
def home():

    return {
        "message": "Traffic Congestion Prediction API"
    }

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "congestion_level": int(prediction[0])
    }