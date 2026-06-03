import streamlit as st
import pandas as pd
import joblib
from src.models.predict import predict_text


st.title("SAR Social Media Analyzer")

MODEL_OPTIONS = {

    "Logistic Regression":
        "models/logistic_regression.pkl",

    "SVM":
        "models/svm_model.pkl",

    "Tuned SVM":
        "models/best_svm_model.pkl",

    "Naive Bayes":
        "models/naive_bayes_model.pkl",

    "Random Forest":
        "models/random_forest.pkl",

    "Tuned Random Forest":
        "models/best_random_forest.pkl"
}

selected_model = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "SVM",
        "Tuned SVM",
        "Naive Bayes",
        "Random Forest",
        "Tuned Random Forest"
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
        "results/model_comparison.csv"
    )

    st.dataframe(results, use_container_width=True)

except:
    st.warning("No comparison results found.")
    
try:

    best_model = results.loc[
        results["F1"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(F1 = {best_model['F1']:.4f})"
    )

except:
    pass   

st.subheader("Performance Metrics")

try:

    metrics_df = results.set_index(
        "Model"
    )

    st.bar_chart(
        metrics_df[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]
        ]
    )

except:
    pass 

st.header("Final Model Evaluation")

try:
    st.subheader("Final Results Table")
    final_results = pd.read_csv("results/final_results_table.csv")
    st.dataframe(final_results, use_container_width=True)

    st.subheader("Model Comparison Based on F1-score")
    st.image("results/model_comparison_f1.png")

    st.subheader("Confusion Matrix - Tuned SVM")
    st.image("results/confusion_matrix_tuned_svm.png")

except Exception as e:
    st.warning("Final evaluation files not found.")
    st.code(str(e))