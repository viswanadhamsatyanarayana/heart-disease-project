# ❤️ AI-Powered Heart Disease Prediction

## 📌 Project Overview
This project is an **AI-powered heart disease prediction system** that uses **machine learning models** to predict whether a patient is at risk of heart disease based on clinical features.  
It also integrates a **workflow automation pipeline** for deployment and future enhancements like email notifications and AI-driven explanations.

---

## ⚙️ Features
- ✅ **Dataset preprocessing** (cleaning & scaling)  
- ✅ **Model training** with Logistic Regression, Random Forest, and XGBoost  
- ✅ **Evaluation** (accuracy, confusion matrix, ROC curve, etc.)  
- ✅ **Best model saved** (`best_model.pkl` + `scaler.pkl`)  
- ✅ **Interactive Gradio app** for predictions  
- 🚀 **Deployable on Hugging Face Spaces**  
- 📧 **Workflow automation (n8n + email)** – *Phase 4*  

---

## 🧑‍⚕️ Input Features
The app takes the following medical parameters as input:

- Age  
- Sex (0 = Female, 1 = Male)  
- Chest Pain Type (0–3)  
- Resting Blood Pressure  
- Serum Cholesterol (mg/dl)  
- Fasting Blood Sugar (>120 mg/dl, 1 = True, 0 = False)  
- Resting ECG Results (0–2)  
- Maximum Heart Rate Achieved  
- Exercise-Induced Angina (1 = Yes, 0 = No)  
- ST Depression (Oldpeak)  
- Slope (0–2)  
- Major Vessels Colored by Fluoroscopy (0–3)  
- Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)  

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/viswanadhamsatyanarayana/heart-disease-project.git
   cd heart-disease-project
