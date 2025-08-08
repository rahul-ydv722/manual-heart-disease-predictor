import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💓 Heart Disease Risk Predictor")

def user_input():
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 200, 150)
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect, 3 = Unknown)", [0, 1, 2, 3])
    
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

input_data = user_input()

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease.")
