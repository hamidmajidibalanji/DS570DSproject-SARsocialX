import streamlit as st
import pandas as pd
import joblib
from src.models.predict import predict_text


st.title("SAR Social Media Analyzer")

MODEL_OPTIONS = {
    "Tuned SVM": "models/best_svm_model.pkl",
    "SVM": "models/svm_model.pkl",
    "Logistic Regression": "models/logistic_regression.pkl"
}

selected_model = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "SVM",
        "Tuned SVM"
    ]
)

tweet = st.text_area(
    "Enter Tweet Text"
)

if st.button("Analyze"):

    prediction = predict_text(
        tweet,
        selected_model
    )

    st.success(
        f"Prediction: {prediction}"
    )
    
st.header("Model Comparison")

try:

    results = pd.read_csv(
        "outputs/model_comparison.csv"
    )

    st.dataframe(results)

except:
    st.warning("No comparison results found.")