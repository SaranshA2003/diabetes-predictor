import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔍 Diabetes Prediction App")

# Get user input
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    # Create data array
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("🔴 The person is likely to have diabetes.")
    else:
        st.success("🟢 The person is likely NOT to have diabetes.")





