import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model & scaler from environment variables
model_path = os.environ.get("MODEL_PATH", "best_model.pkl")
scaler_path = os.environ.get("SCALER_PATH", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API 🚀")

# Define input data structure
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

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running 🚀"}

# Predict endpoint
@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]]

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]  # probability of heart disease

    return {
        "prediction": int(prediction[0]),
        "probability": round(float(probability), 2)
    }

# Optional: Run locally if needed
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
