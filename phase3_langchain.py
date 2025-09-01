# phase3_langchain.py

import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# -------------------------------
# Step 1: ML API endpoint
# -------------------------------
API_URL = "http://127.0.0.1:8000/predict"

# -------------------------------
# Step 2: Function to call ML API
# -------------------------------
def get_prediction(patient_data):
    try:
        response = requests.post(API_URL, json=patient_data)
        if response.status_code == 200:
            return response.json()  # returns {'prediction': 0/1, 'probability': 0.xx}
        else:
            return {"error": f"Failed to get prediction. Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# -------------------------------
# Step 3: LangChain prompt
# -------------------------------
template = """
You are a helpful medical assistant. Given patient data and the ML model prediction, 
generate a clear and concise explanation for a doctor:

Patient Data: {patient_data}
Prediction: {prediction}

Explain the result in simple terms and provide insights about risk factors.
"""

prompt = PromptTemplate(
    input_variables=["patient_data", "prediction"],
    template=template
)

# -------------------------------
# Step 4: Setup LLM (OpenAI)
# -------------------------------
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

# -------------------------------
# Step 5: Example patient
# -------------------------------
patient_example = {
    "age": 60,
    "sex": 1,
    "cp": 3,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

# -------------------------------
# Step 6: Run prediction & explanation
# -------------------------------
prediction = get_prediction(patient_example)
if "error" not in prediction:
    explanation = chain.run(patient_data=patient_example, prediction=prediction)
    print("ML Prediction:", prediction)
    print("Explanation for Doctor:\n", explanation)
else:
    print("Error:", prediction["error"])
