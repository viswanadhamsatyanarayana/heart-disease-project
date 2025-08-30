from fastapi import FastAPI
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("../model/best_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: dict):
    # Example input: {"age": 55, "chol": 200, ...}
    features = np.array([list(data.values())]).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return {"prediction": int(prediction[0])}
