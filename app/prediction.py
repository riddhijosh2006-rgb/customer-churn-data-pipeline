"""
ML Prediction page.

Keeps the original prediction form and behavior from the initial
app.py, wrapped as a page function, with added risk-level labelling
and error handling if the model or its inputs are missing/broken.
"""

import os

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import config

# Risk thresholds used to translate a churn probability into a label.
# Documented here since the dashboard displays them to the user.
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60


@st.cache_resource
def load_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{config.MODEL_PATH}'. "
            "Run src/train_final.py to train and save the model first."
        )
    return joblib.load(config.MODEL_PATH)


def risk_level(probability: float) -> tuple[str, str]:
    """Map a churn probability to a (label, streamlit-style) pair."""
    if probability >= MEDIUM_RISK_MAX:
        return "High Risk", "error"
    if probability >= LOW_RISK_MAX:
        return "Medium Risk", "warning"
    return "Low Risk", "success"


def render():
    st.title("🤖 ML Prediction")
    st.markdown(
        "Enter a customer's information below and the trained churn "
        "pipeline (`models/churn_pipeline.joblib`) will estimate their "
        "churn probability."
    )
    st.caption(
        f"Risk thresholds: **Low** < {LOW_RISK_MAX:.0%}, "
        f"**Medium** {LOW_RISK_MAX:.0%}\u2013{MEDIUM_RISK_MAX:.0%}, "
        f"**High** \u2265 {MEDIUM_RISK_MAX:.0%}."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.divider()
    st.subheader("👤 Customer Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])

    with col2:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col3:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    st.subheader("🌐 Internet & Services")
    col1, col2, col3 = st.columns(3)

    with col1:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("💳 Billing Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col3:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    st.divider()

    if st.button("🔮 Predict Churn", use_container_width=True):
        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        })

        # Guard against schema drift between the form and the trained model.
        missing = set(config.MODEL_FEATURE_COLUMNS) - set(input_data.columns)
        if missing:
            st.error(f"Input is missing columns the model expects: {missing}")
            return

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            return

        st.subheader("📊 Prediction Result")

        label, _ = risk_level(probability)

        if prediction == "Yes":
            st.error("🚨 Predicted to churn")
        else:
            st.success("✅ Predicted to stay")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
        with col2:
            st.metric("Risk Level", label)
