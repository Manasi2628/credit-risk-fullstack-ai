from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Credit Risk Scoring API")

# Load model & scaler
model = joblib.load("model/credit_model.pkl")
scaler = joblib.load("model/scaler.pkl")

class Applicant(BaseModel):
    features: list

@app.post("/predict")
def predict(applicant: Applicant):
    data = np.array(applicant.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    return {"prediction": int(prediction), "risk_score": float(probability)}
