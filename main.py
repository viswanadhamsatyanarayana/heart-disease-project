import joblib
import numpy as np
import gradio as gr

# Load model & scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal):
    try:
        # Prepare input
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])

        # Scale
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        result = "⚠️ Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease"
        return f"{result}\n\nProbability: {round(probability*100, 2)} %"
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
    gr.Number(label="Chest Pain Type"),
    gr.Number(label="Resting BP"),
    gr.Number(label="Cholesterol"),
    gr.Radio([0, 1], label="Fasting Blood Sugar > 120mg/dl"),
    gr.Number(label="Rest ECG"),
    gr.Number(label="Max Heart Rate"),
    gr.Radio([0, 1], label="Exercise Induced Angina"),
    gr.Number(label="Oldpeak (ST Depression)"),
    gr.Number(label="Slope of Peak Exercise ST Segment"),
    gr.Number(label="Number of Major Vessels (0-3)"),
    gr.Number(label="Thal (0=Normal, 1=Fixed defect, 2=Reversible defect)")
]

outputs = gr.Textbox(label="Prediction Result")

iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="💓 Heart Disease Prediction AI",
    description="Enter patient data to predict the risk of heart disease."
)

if __name__ == "__main__":
    iface.launch()
