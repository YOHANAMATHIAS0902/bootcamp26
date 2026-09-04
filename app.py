import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("student_dropout.csv")

df = load_data()

# Sidebar
st.sidebar.title("📊 Dashboard Menu")
page = st.sidebar.radio("Navigate", ["Overview", "Data", "Visualization"])

# =======================
# 🏠 OVERVIEW PAGE
# =======================
if page == "Overview":
    st.title("🎓 Student Dropout Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))

    if 'target' in df.columns:
        col2.metric("Dropouts", (df['target'] == 'dropout').sum())
        col3.metric("Graduates", (df['target'] == 'graduate').sum())

    st.write("### Dataset Preview")
    st.dataframe(df.head())

# =======================
# 📂 DATA PAGE
# =======================
elif page == "Data":
    st.title("📂 Full Dataset")

    st.dataframe(df)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

# =======================
# 📈 VISUALIZATION PAGE
# =======================
elif page == "Visualization":
    st.title("📈 Data Visualization")

    if 'target' in df.columns:
        st.subheader("Target Distribution")
        st.bar_chart(df['target'].value_counts())

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=False, ax=ax)
        st.pyplot(fig)
