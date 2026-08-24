import streamlit as st
import pandas as pd
import joblib
import os
from src.ui import load_css

st.set_page_config(
    page_title="Batch Fraud Prediction",
    page_icon="📁",
    layout="wide"
)
load_css()
st.title("📁 Batch Fraud Prediction")
st.write(
    "Upload a CSV file containing multiple transactions "
    "and predict fraud for all transactions."
)


@st.cache_resource
def load_model():
    model_path = "models/fraud_model.pkl"

    if not os.path.exists(model_path):
        return None

    return joblib.load(model_path)


model_data = load_model()

if model_data is None:
    st.error(
        "Model not found. Please run model training first."
    )
    st.stop()


model = model_data["model"]
features = model_data["features"]

st.success(
    f"Model loaded successfully with {len(features)} features."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Dataset Preview")

        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )

        st.subheader("🔍 Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Rows",
            len(uploaded_df)
        )

        col2.metric(
            "Total Columns",
            len(uploaded_df.columns)
        )

        available_features = [
            feature
            for feature in features
            if feature in uploaded_df.columns
        ]

        missing_features = [
            feature
            for feature in features
            if feature not in uploaded_df.columns
        ]

        col3.metric(
            "Required Features Found",
            f"{len(available_features)}/{len(features)}"
        )

        if missing_features:

            st.warning(
                "Some required features are missing:"
            )

            st.write(missing_features)

        else:

            st.success(
                "All required model features are available!"
            )

            if st.button(
                "🚀 Run Batch Fraud Detection",
                use_container_width=True
            ):

                prediction_data = uploaded_df[
                    features
                ].copy()

                # Convert to numeric
                for feature in features:

                    prediction_data[feature] = pd.to_numeric(
                        prediction_data[feature],
                        errors="coerce"
                    )

                    median_value = prediction_data[
                        feature
                    ].median()

                    prediction_data[feature] = (
                        prediction_data[feature]
                        .fillna(median_value)
                    )

                predictions = model.predict(
                    prediction_data
                )

                probabilities = model.predict_proba(
                    prediction_data
                )

                result_df = uploaded_df.copy()

                result_df[
                    "Fraud_Prediction"
                ] = [
                    "Fraud"
                    if prediction == 1
                    else "Legitimate"
                    for prediction in predictions
                ]

                result_df[
                    "Fraud_Probability"
                ] = (
                    probabilities[:, 1] * 100
                )

                st.divider()

                st.subheader(
                    "🛡️ Prediction Results"
                )

                fraud_count = (
                    predictions == 1
                ).sum()

                legitimate_count = (
                    predictions == 0
                ).sum()

                fraud_rate = (
                    fraud_count / len(predictions) * 100
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    len(predictions)
                )

                col2.metric(
                    "🚨 Fraud Detected",
                    fraud_count
                )

                col3.metric(
                    "📉 Fraud Rate",
                    f"{fraud_rate:.2f}%"
                )

                st.divider()

                st.subheader(
                    "📋 Detailed Prediction Results"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as error:

        st.error(
            f"Error processing file: {error}"
        )