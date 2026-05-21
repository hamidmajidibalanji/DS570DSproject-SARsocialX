import sys
import os

sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd

from src.models.predict import predict_tweet


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SAR Social Media Analyzer",
    page_icon="🚨",
    layout="wide"
)


# --------------------------------------------------
# Title and Description
# --------------------------------------------------

st.title("🚨 Search and Rescue (SAR) Social Media Analyzer")

st.markdown("""
This system analyzes social media posts and classifies them as:

- **Disaster / Emergency Related**
- **Non-Disaster**

The model is based on:
- TF-IDF feature extraction
- Logistic Regression classification
- Disaster tweet datasets
""")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.info("""
DS570 Project

Features:
- Tweet preprocessing
- NLP classification
- TF-IDF vectorization
- Logistic Regression
- Streamlit dashboard
""")


# --------------------------------------------------
# Tweet Prediction Section
# --------------------------------------------------

st.header("🔍 Tweet Classification")

user_input = st.text_area(
    "Enter a tweet or social media post:",
    height=150,
    placeholder="Example: People are trapped under collapsed buildings and need urgent rescue."
)


if st.button("Analyze Tweet"):

    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:

        prediction = predict_tweet(user_input)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Disaster / Emergency Tweet Detected")
        else:
            st.success("✅ Non-Disaster Tweet")


# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.header("📊 Dataset Preview")

try:

    df = pd.read_csv(
        "data/processed/processed_disaster_tweets.csv"
    )

    st.write("Processed Dataset Sample:")

    st.dataframe(df.head(10))

    st.write(f"Dataset Size: {len(df)} tweets")

except Exception as e:

    st.warning("Processed dataset not found.")

    st.code(str(e))


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.header("🧠 Model Information")

st.markdown("""
### Machine Learning Pipeline

1. Text Cleaning
2. TF-IDF Feature Extraction
3. Logistic Regression Classification
4. Disaster Tweet Prediction

### Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
""")


# --------------------------------------------------
# Confusion Matrix Display
# --------------------------------------------------

st.header("📈 Evaluation Visualization")

try:

    st.image(
        "outputs/confusion_matrix.png",
        caption="Confusion Matrix"
    )

except:
    st.info("Confusion matrix image not found yet.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "DS570 - Search and Rescue Social Media Analysis System"
)