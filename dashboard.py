import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/fraud_model.pkl")

df = load_data()
model = load_model()

# Prediction
df['Prediction'] = model.predict(df.drop(columns=['Class'], errors='ignore'))

# Sidebar filters
st.sidebar.subheader("Filter")
filter_option = st.sidebar.radio("Show:", ['All', 'Only Fraud', 'Only Legitimate'])
if filter_option == 'Only Fraud':
    df = df[df['Prediction'] == 1]
elif filter_option == 'Only Legitimate':
    df = df[df['Prediction'] == 0]

# Display metrics
st.title("🔍 Fraud Detection Dashboard")
st.metric("Total Transactions", len(df))
st.metric("Predicted Fraudulent", df['Prediction'].sum())

# Bar chart
st.subheader("Prediction Count")
fig, ax = plt.subplots()
sns.countplot(x='Prediction', data=df, ax=ax)
st.pyplot(fig)

# Show preview of data
st.subheader("Preview of Transactions")
st.dataframe(df.head(50))
