import streamlit as st
import numpy as np
from diabetes_model import load_data, train_models

# Load data and train models
x_train, x_test, y_train, y_test = load_data()
models = train_models(x_train, x_test, y_train, y_test)

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("Provide the patient's medical details below to predict diabetes:")

# Input form
preg = st.number_input("Pregnancies", min_value=0, step=1)
gluc = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

user_input = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])

model_choice = st.selectbox("Choose Model", ["KNN", "Decision Tree", "MLP Neural Net"])

if st.button("Predict"):
    if model_choice == "MLP Neural Net":
        user_input_scaled = models["scaler"].transform(user_input)
        result = models["mlp"].predict(user_input_scaled)
    elif model_choice == "KNN":
        result = models["knn"].predict(user_input)
    else:
        result = models["tree"].predict(user_input)

    st.success("Prediction: " + ("🟢 Non-Diabetic" if result[0] == 0 else "🔴 Diabetic"))
