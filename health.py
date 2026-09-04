import joblib
import pandas as pd
import streamlit as st
import datetime

st.set_page_config(page_title="Patient Disease Risk Predictor", page_icon="🩺")

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

st.title("🩺 Patient Disease Risk Predictor")
st.caption("Predicts Has_Disease from patient health record fields.")


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def age_group(age):
    if age < 18:
        return "0-17"
    if age < 35:
        return "18-34"
    if age < 50:
        return "35-49"
    if age < 65:
        return "50-64"
    return "65+"


with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=110, value=45)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
        systolic = st.number_input("Systolic BP", min_value=70, max_value=220, value=120)
        diastolic = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=400, value=190)
        follow_up_days = st.number_input("Follow-Up (days)", min_value=0, max_value=90, value=30)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
        city = st.selectbox("City", ["New York", "Chicago", "Los Angeles", "Boston", "Unknown"])
        diabetic = st.selectbox("Diabetic", ["Yes", "No", "Unknown"])
        smoker = st.selectbox("Smoker", ["Yes", "No", "Former", "Unknown"])
        diagnosis_code = st.selectbox("Diagnosis Code", ["A00", "B20", "E11.9", "I10", "Unknown"])
        medication_count = st.number_input("Number of Medications", min_value=0, max_value=10, value=0)
        has_notes = st.checkbox("Has clinical notes on file", value=True)
        notes_length = st.number_input("Notes length (characters)", min_value=0, max_value=500, value=40)
        days_since_visit = st.number_input("Days since last visit", min_value=0, max_value=3650, value=180)

    submitted = st.form_submit_button("Predict")


if submitted:
    chol_band = "Normal" if cholesterol < 200 else ("Borderline" if cholesterol < 240 else "High")

    row = pd.DataFrame([{
        "Age": age,
        "Heart_Rate": heart_rate,
        "BMI": bmi,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Pulse_Pressure": systolic - diastolic,
        "Cholesterol_Numeric": cholesterol,
        "Follow_Up_Days": follow_up_days,
        "Medication_Count": medication_count,
        "Has_Medication": int(medication_count > 0),
        "Has_Notes": int(has_notes),
        "Notes_Length": notes_length,
        "Days_Since_Last_Visit": days_since_visit,
        "Was_Missing_Age": 0,
        "Was_Missing_Heart_Rate": 0,
        "Was_Missing_BMI": 0,
        "Was_Missing_Systolic_BP": 0,
        "Was_Missing_Cholesterol_Numeric": 0,
        "Was_Missing_Follow_Up_Days": 0,
        "BMI_Category": bmi_category(bmi),
        "Age_Group": age_group(age),
        "Gender": gender,
        "City": city,
        "Diabetic": diabetic,
        "Smoker": smoker,
        "Cholesterol_Band": chol_band,
        "Diagnosis_Code": diagnosis_code,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0, 1]

    st.subheader("Result")
    if pred == 1:
        st.error(f"Prediction: Has disease (probability = {proba:.2%})")
    else:
        st.success(f"Prediction: No disease (probability = {proba:.2%})")