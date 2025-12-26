import streamlit as st
import numpy as np
import pickle
import time

# Page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Load model and scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------- UI HEADER --------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.result-good {
    background-color: #0f5132;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 22px;
}
.result-bad {
    background-color: #842029;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Enter patient details to check diabetes risk</p>",
    unsafe_allow_html=True
)

st.write("")

# -------------------- INPUT CARD --------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, step=1)
    glucose = st.number_input("🧪 Glucose Level", min_value=0)
    blood_pressure = st.number_input("💓 Blood Pressure", min_value=0)
    bmi = st.number_input("⚖️ BMI", min_value=0.0, format="%.1f")

with col2:
    skin_thickness = st.number_input("📏 Skin Thickness", min_value=0)
    insulin = st.number_input("💉 Insulin Level", min_value=0)
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("🎂 Age", min_value=0, step=1)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- PREDICT BUTTON --------------------
predict_btn = st.button("🔍 Predict Diabetes")

if predict_btn:
    with st.spinner("Analyzing data..."):
        time.sleep(1.5)

        input_data = np.array([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

    # -------------------- RESULT PANEL --------------------
    if prediction == 1:
        st.markdown(
            "<div class='result-bad'>❌ High Risk: Person is Diabetic</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-good'>✅ Low Risk: Person is Not Diabetic</div>",
            unsafe_allow_html=True
        )

# -------------------- FOOTER --------------------
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>ML Project | Built with Streamlit</p>",
    unsafe_allow_html=True
)
