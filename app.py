import streamlit as st
import pandas as pd
import joblib
import os

from src.data_preprocessing import load_and_merge_data, clean_data
from src.ui import load_css
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
    }

    .feature-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = load_and_merge_data()
    return clean_data(df)


@st.cache_resource
def load_model():
    model_path = "models/fraud_model.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)

    return None


try:
    df = load_data()
except Exception as error:
    df = None
    st.error(f"Dataset loading error: {error}")

model_data = load_model()

st.markdown(
    '<div class="main-title">🛡️ FraudGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Fraud Detection & Financial Transaction Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

if df is not None:

    total_transactions = len(df)

    target_column = None

    if model_data:
        target_column = model_data.get("target_column")

    fraud_count = 0
    fraud_rate = 0

    if target_column in df.columns:
        fraud_count = int(
            pd.to_numeric(
                df[target_column],
                errors="coerce"
            ).fillna(0).sum()
        )

        fraud_rate = (
            fraud_count / total_transactions * 100
            if total_transactions > 0
            else 0
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📊 Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "🚨 Fraud Cases",
        f"{fraud_count:,}"
    )

    col3.metric(
        "📉 Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    if model_data:
        accuracy = model_data.get("accuracy", 0)

        col4.metric(
            "🤖 Model Accuracy",
            f"{accuracy:.2%}"
        )
    else:
        col4.metric(
            "🤖 Model Status",
            "Not Loaded"
        )

st.divider()

st.subheader("✨ Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Smart Analytics
    Explore transaction patterns, missing values,
    fraud statistics and financial insights.
    """)

with col2:
    st.markdown("""
    ### 🤖 Machine Learning
    Random Forest model trained to identify
    suspicious financial transactions.
    """)

with col3:
    st.markdown("""
    ### 🔍 Fraud Prediction
    Enter transaction details and receive
    instant fraud probability and risk level.
    """)

st.divider()

st.subheader("📌 How to Use")

st.markdown("""
1. **Data Analysis** → Explore your dataset.
2. **Visualizations** → Discover fraud patterns.
3. **Model Performance** → Evaluate the ML model.
4. **Fraud Prediction** → Test new transactions.
5. **Batch Prediction** → Upload a CSV and detect fraud in multiple transactions.
6. Use the sidebar to navigate between all pages.
""")

if df is not None:
    with st.expander("🔍 Preview Transaction Dataset"):
        st.dataframe(
            df.head(10),
            use_container_width=True
        )