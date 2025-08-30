import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model & scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Heart Disease Prediction API")

class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]]

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]  # risk of disease

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
