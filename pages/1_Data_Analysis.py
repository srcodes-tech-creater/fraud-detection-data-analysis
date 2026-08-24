import streamlit as st
import pandas as pd

from src.data_preprocessing import load_and_merge_data, clean_data

from src.ui import load_css
st.set_page_config(
    page_title="Data Analysis",
    page_icon="📊",
    layout="wide"
)
load_css()

@st.cache_data
def load_data():
    df = load_and_merge_data()
    df = clean_data(df)
    return df


df = load_data()

st.title("📊 Data Analysis")

st.subheader("Dataset Information")

col1, col2 = st.columns(2)

col1.metric(
    "Rows",
    df.shape[0]
)

col2.metric(
    "Columns",
    df.shape[1]
)


st.divider()

st.subheader("📋 Column Names")

st.write(
    df.columns.tolist()
)


st.divider()

st.subheader("🔍 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)


st.divider()

st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)


st.divider()

st.subheader("❓ Missing Values")

missing_values = df.isnull().sum()

missing_df = pd.DataFrame(
    {
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    }
)

st.dataframe(
    missing_df,
    use_container_width=True
)


st.divider()

st.subheader("🔎 Data Types")

data_types = pd.DataFrame(
    {
        "Column": df.dtypes.index,
        "Data Type": df.dtypes.astype(str).values
    }
)

st.dataframe(
    data_types,
    use_container_width=True
)