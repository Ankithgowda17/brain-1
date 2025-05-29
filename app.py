import streamlit as st
import numpy as np
import pickle

# Load models
rf_model = pickle.load(open('model1.pkl', 'rb'))
xgb_model = pickle.load(open('model2.pkl', 'rb'))
meta_model = pickle.load(open('model3.pkl', 'rb'))

st.title("🧠 Brain Stroke Risk Predictor")

# Input form
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
alcohol_intake = st.selectbox("Alcohol Intake", ["none", "occasional", "regular"])
diet_type = st.selectbox("Diet Type", ["healthy", "average", "unhealthy"])
sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
stroke_family_history = st.selectbox("Stroke Family History", [0, 1])
salt_intake = st.selectbox("Salt Intake", ["low", "medium", "high"])
systolic_bp = st.number_input("Systolic BP", value=120.0)
diastolic_bp = st.number_input("Diastolic BP", value=80.0)
ldl = st.number_input("LDL Cholesterol", value=130.0)
hdl = st.number_input("HDL Cholesterol", value=40.0)

# Encodings for categorical inputs
gender_map = {"Male": 1, "Female": 0, "Other": 2}
ever_married_map = {"Yes": 1, "No": 0}
work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
res_map = {"Urban": 1, "Rural": 0}
smoke_map = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}
alcohol_map = {"none": 0, "occasional": 1, "regular": 2}
diet_map = {"healthy": 0, "average": 1, "unhealthy": 2}
sleep_map = {"poor": 0, "average": 1, "good": 2}
salt_map = {"low": 0, "medium": 1, "high": 2}

# Prepare input data
input_data = np.array([[
    gender_map[gender],
    age,
    hypertension,
    heart_disease,
    ever_married_map[ever_married],
    work_map[work_type],
    res_map[residence_type],
    avg_glucose_level,
    bmi,
    smoke_map[smoking_status],
    alcohol_map[alcohol_intake],
    diet_map[diet_type],
    sleep_map[sleep_quality],
    stroke_family_history,
    salt_map[salt_intake],
    systolic_bp,
    diastolic_bp,
    ldl,
    hdl
]])

# Base predictions
rf_pred = rf_model.predict_proba(input_data)[:, 1]
xgb_pred = xgb_model.predict_proba(input_data)[:, 1]

# Meta model prediction
stacked = np.column_stack((rf_pred, xgb_pred))
final_pred = meta_model.predict(stacked)[0]
final_prob = meta_model.predict_proba(stacked)[0][1]

# Display result
if final_pred == 1:
    st.error(f"⚠️ High Stroke Risk (Probability: {final_prob:.2f})")
else:
    st.success(f"✅ Low Stroke Risk (Probability: {final_prob:.2f})")
