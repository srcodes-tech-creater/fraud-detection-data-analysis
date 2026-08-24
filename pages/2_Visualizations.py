import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_preprocessing import (
    load_and_merge_data,
    clean_data
)

from src.ui import load_css
st.set_page_config(
    page_title="Visualizations",
    page_icon="📈",
    layout="wide"
)
load_css()

@st.cache_data
def load_data():

    df = load_and_merge_data()

    return clean_data(df)


df = load_data()

st.title("📈 Interactive Fraud Analytics")


fraud_columns = [
    "FraudIndicator",
    "Fraud",
    "IsFraud",
    "is_fraud"
]

target = next(
    (
        column
        for column in fraud_columns
        if column in df.columns
    ),
    None
)


if target:

    st.subheader(
        "🚨 Fraud Distribution"
    )

    counts = (
        df[target]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Transaction Type",
        "Count"
    ]

    counts["Transaction Type"] = (
        counts["Transaction Type"]
        .map({
            0: "Legitimate",
            1: "Fraud"
        })
        .fillna(
            counts["Transaction Type"]
            .astype(str)
        )
    )

    fig = px.pie(
        counts,
        names="Transaction Type",
        values="Count",
        title="Fraud vs Legitimate Transactions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


if numeric_columns:

    st.divider()

    st.subheader(
        "📊 Feature Distribution Explorer"
    )

    selected_feature = st.selectbox(
        "Select a Numeric Feature",
        numeric_columns
    )

    fig = px.histogram(
        df,
        x=selected_feature,
        nbins=40,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


if len(numeric_columns) >= 2:

    st.divider()

    st.subheader(
        "🔗 Feature Relationship Explorer"
    )

    col1, col2 = st.columns(2)

    x_feature = col1.selectbox(
        "X-axis",
        numeric_columns,
        index=0
    )

    y_feature = col2.selectbox(
        "Y-axis",
        numeric_columns,
        index=1
    )

    color_feature = (
        target
        if target
        else None
    )

    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=color_feature,
        title=f"{x_feature} vs {y_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )