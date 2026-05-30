import streamlit as st
import pandas as pd
import joblib

st.title("SAR Social Media Analyzer")

MODEL_OPTIONS = {
    "Tuned SVM": "models/best_svm_model.pkl",
    "SVM": "models/svm_model.pkl",
    "Logistic Regression": "models/logistic_regression.pkl"
}

selected_model = st.selectbox(
    "Choose Model",
    list(MODEL_OPTIONS.keys())
)

tweet = st.text_area(
    "Enter Tweet Text"
)

if st.button("Analyze"):

    model = joblib.load(
        MODEL_OPTIONS[selected_model]
    )

    prediction = model.predict([tweet])[0]

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