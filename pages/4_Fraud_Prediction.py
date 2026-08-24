import streamlit as st
import joblib
import pandas as pd
import os


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
from src.ui import load_css
st.set_page_config(
    page_title="Fraud Prediction",
    page_icon="🔍",
    layout="wide"
)
load_css()

st.title("🔍 AI Fraud Prediction")
st.write(
    "Enter transaction details to predict fraud risk using the trained Machine Learning model."
)


# -----------------------------------------
# FILE PATHS
# -----------------------------------------

MODEL_PATH = "models/fraud_model.pkl"

DATA_PATH = "data/processed/merged_fraud_data.csv"


# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


# -----------------------------------------
# LOAD DATA
# -----------------------------------------

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


# -----------------------------------------
# LOAD FILES
# -----------------------------------------

model_data = load_model()

training_data = load_data()


# -----------------------------------------
# CHECK MODEL
# -----------------------------------------

if model_data is None:

    st.error(
        "🚨 Model not found!"
    )

    st.code(
        "python src/model_training.py"
    )

    st.stop()


# -----------------------------------------
# MODEL INFORMATION
# -----------------------------------------

model = model_data["model"]

features = model_data["features"]

threshold = model_data.get(
    "threshold",
    0.5
)


st.success(
    f"🤖 AI Model Loaded Successfully | "
    f"{len(features)} Features"
)


st.info(
    f"🎯 Fraud Detection Threshold: {threshold:.2f}"
)


st.divider()


# -----------------------------------------
# TRANSACTION INPUT
# -----------------------------------------

st.subheader(
    "📝 Enter Transaction Details"
)


inputs = {}


# Create two columns
left_column, right_column = st.columns(2)


for index, feature in enumerate(features):

    default_value = 0.0

    min_value = 0.0

    max_value = None


    # Get useful default values
    if (
        training_data is not None
        and feature in training_data.columns
    ):

        feature_data = pd.to_numeric(
            training_data[feature],
            errors="coerce"
        )

        median_value = feature_data.median()

        if pd.notna(median_value):

            default_value = float(
                median_value
            )

        minimum = feature_data.min()

        maximum = feature_data.max()

        if pd.notna(minimum):

            min_value = float(minimum)

        if pd.notna(maximum):

            max_value = float(maximum)


    # Select column
    column = (
        left_column
        if index % 2 == 0
        else right_column
    )


    with column:

        if max_value is not None:

            inputs[feature] = st.number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=default_value
            )

        else:

            inputs[feature] = st.number_input(
                feature,
                value=default_value
            )


st.divider()


# -----------------------------------------
# PREDICT BUTTON
# -----------------------------------------

if st.button(
    "🛡️ Analyze Fraud Risk",
    use_container_width=True
):

    # Create DataFrame
    input_df = pd.DataFrame(
        [inputs],
        columns=features
    )


    # -----------------------------------------
    # PREDICTION PROBABILITY
    # -----------------------------------------

    probabilities = model.predict_proba(
        input_df
    )


    fraud_probability = (
        probabilities[0][1]
    )


    # -----------------------------------------
    # APPLY THRESHOLD
    # -----------------------------------------

    prediction = int(
        fraud_probability >= threshold
    )


    fraud_percentage = (
        fraud_probability * 100
    )


    legitimate_percentage = (
        100 - fraud_percentage
    )


    # -----------------------------------------
    # RISK LEVEL
    # -----------------------------------------

    if fraud_percentage >= 75:

        risk_level = "🔴 HIGH RISK"

        risk_message = (
            "This transaction shows strong indicators of potential fraud."
        )

    elif fraud_percentage >= 40:

        risk_level = "🟠 MEDIUM RISK"

        risk_message = (
            "This transaction contains suspicious patterns and should be reviewed."
        )

    else:

        risk_level = "🟢 LOW RISK"

        risk_message = (
            "This transaction appears to have a low fraud risk."
        )


    # -----------------------------------------
    # RESULT
    # -----------------------------------------

    st.divider()

    st.subheader(
        "📊 Fraud Analysis Result"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Fraud Probability",
        f"{fraud_percentage:.2f}%"
    )


    col2.metric(
        "Legitimate Probability",
        f"{legitimate_percentage:.2f}%"
    )


    col3.metric(
        "Risk Level",
        risk_level
    )


    # -----------------------------------------
    # ALERT
    # -----------------------------------------

    if prediction == 1:

        st.error(
            "🚨 FRAUD ALERT: "
            "This transaction is predicted as potentially fraudulent."
        )

    else:

        st.success(
            "✅ TRANSACTION APPROVED: "
            "This transaction is predicted as legitimate."
        )


    st.info(
        f"💡 {risk_message}"
    )


    # -----------------------------------------
    # PROBABILITY PROGRESS BAR
    # -----------------------------------------

    st.subheader(
        "📈 Fraud Risk Probability"
    )

    st.progress(
        int(fraud_percentage)
    )

    st.write(
        f"Fraud Probability: **{fraud_percentage:.2f}%**"
    )


    # -----------------------------------------
    # RESULT TABLE
    # -----------------------------------------

    result_df = input_df.copy()

    result_df["Fraud_Probability"] = (
        f"{fraud_percentage:.2f}%"
    )

    result_df["Prediction"] = (
        "Fraud"
        if prediction == 1
        else "Legitimate"
    )

    result_df["Risk_Level"] = (
        risk_level
    )


    st.subheader(
        "📋 Prediction Details"
    )


    st.dataframe(
        result_df,
        use_container_width=True
    )


    # -----------------------------------------
    # DOWNLOAD RESULTS
    # -----------------------------------------

    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="fraud_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# -----------------------------------------
# INFORMATION SECTION
# -----------------------------------------

st.divider()

st.subheader(
    "ℹ️ About the Prediction"
)

st.write(
    """
    The system uses the trained Random Forest Machine Learning model
    to analyze transaction features. The fraud probability is calculated
    using the model, and the saved fraud detection threshold determines
    the final prediction.
    """
)