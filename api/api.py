from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

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