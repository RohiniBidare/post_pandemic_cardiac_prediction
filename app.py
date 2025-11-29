# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load trained models and mapping

clf = pickle.load(open("clf_pipeline.pkl", "rb"))   # Classification model
reg = pickle.load(open("reg_pipeline.pkl", "rb"))   # Regression model
disease_map = pickle.load(open("disease_map.pkl", "rb"))  # Mapping dictionary


# Streamlit Page Configuration

st.set_page_config(
    page_title="Post-Pandemic Cardiac Predictor",
    page_icon="🫀",
    layout="centered",
)


# Custom CSS Styling

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #141E30, #243B55);
            color: #fff;
        }
        .main {
            background-color: rgba(0,0,0,0);
        }
        .stApp {
            background: linear-gradient(145deg, #1d2b64, #f8cdda);
        }
        .title {
            font-size: 40px;
            text-align: center;
            color: #F8F8F8;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 5px #000;
        }
        .subtitle {
            text-align: center;
            color: #FFD700;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .result-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        }
        .low {
            background: linear-gradient(135deg, #56ab2f, #a8e063);
        }
        .moderate {
            background: linear-gradient(135deg, #f7971e, #ffd200);
        }
        .high {
            background: linear-gradient(135deg, #cb2d3e, #ef473a);
        }
        .disease-card {
            background: linear-gradient(135deg, #43cea2, #185a9d);
        }
        .mortality-card {
            background: linear-gradient(135deg, #ff6a00, #ee0979);
        }
    </style>
""", unsafe_allow_html=True)

# Header

st.markdown("<div class='title'> Post-Pandemic Cardiac Disease & Mortality Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered prediction system using post-pandemic cardiac health data</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Inputs

st.sidebar.header("‍ Patient Information")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=40)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])

cp_type = st.sidebar.selectbox(
    "Chest Pain Type",
    (
        "1 - Typical Angina: ",
        "2 - Atypical Angina: ",
        "3 - Non-Anginal Pain: ",
        "4 - Asymptomatic:"
    )
)
cp = int(cp_type.split(" - ")[0])

trestbps = st.sidebar.number_input("Resting BP (trestbps)", value=130)
chol = st.sidebar.number_input("Cholesterol (chol)", value=240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])

thalach = st.sidebar.number_input("Max Heart Rate (thalach)", value=150)

exang_status = st.sidebar.radio(
    "🏋️‍♀️ Exercise Induced Angina",
    ["No - No chest pain during exercise", "Yes - Chest pain during exercise"]
)
exang = 1 if "Yes" in exang_status else 0

oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope (1-3)", [1, 2, 3])

ca_type = st.sidebar.selectbox(
    " Major Vessels (ca) — Number of Blocked Coronary Arteries",
    (
        "0 - Healthy",
        "1 - Mild disease",
        "2 - Moderate disease",
        "3 - Severe disease",
        "4 - Critical condition"
    )
)
ca = int(ca_type.split(" - ")[0])

thal_type = st.sidebar.selectbox(
    " Thalassemia Test Result (Thal)",
    (
        "Normal: Normal blood flow (no defect)",
        "Fixed Defect: Damaged heart tissue (permanent)",
        "Reversible Defect: Temporary stress-related blockage"
    )
)

thal_map = {
    "Normal: Normal blood flow (no defect)": 3,
    "Fixed Defect: Damaged heart tissue (permanent)": 6,
    "Reversible Defect: Temporary stress-related blockage": 7
}

thal = thal_map[thal_type]

covid1_status = st.sidebar.radio("COVID-19 Vaccine Dose 1 Taken?", ["Yes", "No"])
covid2_status = st.sidebar.radio("COVID-19 Vaccine Dose 2 Taken?", ["Yes", "No"])

covid1 = 1 if covid1_status == "Yes" else 0
covid2 = 1 if covid2_status == "Yes" else 0

sex_val = 1 if sex == "Male" else 0


# Input DataFrame

input_df = pd.DataFrame([{
    "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
    "thal": thal, "covid1": covid1, "covid2": covid2
}])

st.markdown("###  Input Summary")
st.dataframe(input_df.style.highlight_max(axis=0, color="#ffd700"))


# Predict Button

if st.button("Predict Cardiac Outcome"):

    class_pred = clf.predict(input_df)[0]
    disease_name = disease_map.get(class_pred, f"Unknown ({class_pred})")
    mortality_pred = reg.predict(input_df)[0]

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(input_df)[0]
        confidence = np.max(probs) * 100
    else:
        confidence = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class='result-card disease-card'>
                <h3>🫀 Disease Prediction</h3>
                <h2>{disease_name}</h2>
                {'<p>Confidence: {:.1f}%</p>'.format(confidence) if confidence else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if mortality_pred < 10:
            risk_class = "low"
            risk_text = "Low Risk"
        elif mortality_pred < 30:
            risk_class = "moderate"
            risk_text = "Moderate Risk"
        else:
            risk_class = "high"
            risk_text = "High Risk"

        st.markdown(
            f"""
            <div class='result-card mortality-card'>
                <h3>💀 Predicted Mortality Rate</h3>
                <h2>{mortality_pred:.2f}%</h2>
                <div class='result-card {risk_class}'><h4>{risk_text}</h4></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.info("These predictions are AI-based estimates. Please consult a cardiologist for medical guidance.")

# Suggestions Section

from suggestions import get_suggestions

try:
    sugg = get_suggestions(class_pred, mortality_pred)

    st.markdown("##  Recommendations & Health Suggestions")

    html_block = f"""
    <div style='
        padding:20px;
        border-radius:10px;
        background: rgba(255,255,255,0.08);
        margin-bottom:20px;
        border:1px solid rgba(255,255,255,0.2);
    '>
        <h2 style='color:#FFD700; margin-bottom:5px;'>{sugg['title']}</h2>
        <p style='font-size:18px; color:#eee;'>{sugg['summary']}</p>
    </div>
    """

    st.markdown(html_block, unsafe_allow_html=True)

    st.markdown("###  Suggested Actions:")
    for s in sugg["suggestions"]:
        st.markdown(f"- {s}")

except:
    pass
