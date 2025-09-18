import streamlit as st
import pandas as pd
import requests

st.title("🛡️ Credit Risk Scoring Dashboard")

st.write("Upload applicant data to get risk predictions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data:", df.head())

    predictions = []
    for _, row in df.iterrows():
        features = row.values.tolist()
        response = requests.post("http://127.0.0.1:8000/predict", json={"features": features})
        predictions.append(response.json())

    results = pd.DataFrame(predictions)
    final_df = pd.concat([df, results], axis=1)
    st.write("✅ Predictions with Risk Scores:")
    st.dataframe(final_df)
