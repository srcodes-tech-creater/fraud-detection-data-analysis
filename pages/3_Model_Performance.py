import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
from src.ui import load_css
st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)
load_css()

# -----------------------------------------
# PAGE TITLE
# -----------------------------------------

st.title("🤖 Machine Learning Model Performance")

st.write(
    "Evaluate the performance of the Fraud Detection Machine Learning model."
)


# -----------------------------------------
# FILE PATHS
# -----------------------------------------

MODEL_PATH = "models/fraud_model.pkl"

DATA_PATH = (
    "data/processed/merged_fraud_data.csv"
)


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

df = load_data()


# -----------------------------------------
# CHECK MODEL
# -----------------------------------------

if model_data is None:

    st.error(
        "🚨 Model file not found!"
    )

    st.code(
        "python src/model_training.py"
    )

    st.stop()


# -----------------------------------------
# CHECK DATA
# -----------------------------------------

if df is None:

    st.error(
        "🚨 Processed dataset not found!"
    )

    st.code(
        "python src/data_preprocessing.py"
    )

    st.stop()


# -----------------------------------------
# GET MODEL INFORMATION
# -----------------------------------------

model = model_data["model"]

features = model_data["features"]

target_column = model_data["target_column"]

threshold = model_data.get(
    "threshold",
    0.5
)


# -----------------------------------------
# CHECK REQUIRED COLUMNS
# -----------------------------------------

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]


if missing_features:

    st.error(
        "🚨 Some model features are missing from the dataset!"
    )

    st.write(
        missing_features
    )

    st.stop()


if target_column not in df.columns:

    st.error(
        f"🚨 Target column '{target_column}' not found!"
    )

    st.stop()


# -----------------------------------------
# PREPARE DATA
# -----------------------------------------

X = df[features].copy()

y = df[target_column].copy()


# Convert target to numeric if possible
y = pd.to_numeric(
    y,
    errors="coerce"
)


# Remove rows where target is missing
valid_rows = y.notna()

X = X.loc[valid_rows]

y = y.loc[valid_rows].astype(int)


# Fill missing feature values
X = X.fillna(
    X.median()
)


# -----------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------

try:

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

except Exception as error:

    st.error(
        f"Error splitting the dataset: {error}"
    )

    st.stop()


# -----------------------------------------
# GET FRAUD PROBABILITIES
# -----------------------------------------

probabilities = model.predict_proba(
    X_test
)


# Get the probability for the fraud class
fraud_probability = probabilities[:, 1]


# -----------------------------------------
# APPLY SAVED THRESHOLD
# -----------------------------------------

predictions = (
    fraud_probability >= threshold
).astype(int)


# -----------------------------------------
# CALCULATE METRICS
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# -----------------------------------------
# DISPLAY MODEL METRICS
# -----------------------------------------

st.divider()

st.subheader("📊 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)


col2.metric(
    "Precision",
    f"{precision:.2%}"
)


col3.metric(
    "Recall",
    f"{recall:.2%}"
)


col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)


# -----------------------------------------
# THRESHOLD INFORMATION
# -----------------------------------------

st.info(
    f"🎯 Fraud Detection Threshold: "
    f"{threshold:.2f}"
)


# -----------------------------------------
# CONFUSION MATRIX
# -----------------------------------------

st.divider()

st.subheader(
    "🎯 Confusion Matrix"
)


matrix = confusion_matrix(
    y_test,
    predictions
)


fig, ax = plt.subplots(
    figsize=(7, 5)
)


image = ax.imshow(
    matrix
)


plt.colorbar(
    image,
    ax=ax
)


ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    "Fraud Detection Confusion Matrix"
)


ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)


ax.set_xticklabels(
    [
        "Legitimate",
        "Fraud"
    ]
)


ax.set_yticklabels(
    [
        "Legitimate",
        "Fraud"
    ]
)


# Add values inside the matrix
for i in range(
    matrix.shape[0]
):

    for j in range(
        matrix.shape[1]
    ):

        ax.text(
            j,
            i,
            str(matrix[i, j]),
            ha="center",
            va="center"
        )


st.pyplot(fig)


# -----------------------------------------
# CONFUSION MATRIX EXPLANATION
# -----------------------------------------

st.subheader(
    "📖 Confusion Matrix Explanation"
)


col1, col2 = st.columns(2)


with col1:

    st.write(
        f"**True Negatives:** "
        f"{matrix[0, 0]}"
    )

    st.write(
        f"**False Positives:** "
        f"{matrix[0, 1]}"
    )


with col2:

    st.write(
        f"**False Negatives:** "
        f"{matrix[1, 0]}"
    )

    st.write(
        f"**True Positives:** "
        f"{matrix[1, 1]}"
    )


# -----------------------------------------
# CLASSIFICATION REPORT
# -----------------------------------------

st.divider()

st.subheader(
    "📋 Classification Report"
)


report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


st.dataframe(
    report_df,
    use_container_width=True
)


# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------

if hasattr(
    model,
    "feature_importances_"
):

    st.divider()

    st.subheader(
        "⭐ Feature Importance"
    )


    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": (
                model.feature_importances_
            )
        }
    )


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )


    st.dataframe(
        importance_df,
        use_container_width=True
    )


# -----------------------------------------
# MODEL INFORMATION
# -----------------------------------------

st.divider()

st.subheader(
    "🧠 Model Information"
)


info_col1, info_col2, info_col3 = st.columns(3)


info_col1.metric(
    "Total Features",
    len(features)
)


info_col2.metric(
    "Total Test Samples",
    len(X_test)
)


info_col3.metric(
    "Fraud Cases in Test Data",
    int(y_test.sum())
)


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Fraud Detection Data Analysis | "
    "Machine Learning Performance Evaluation"
)