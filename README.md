# SARSocialX

## Search and Rescue (SAR) Social Media Analyzer using Machine Learning

---

# Overview

SARSocialX is an end-to-end machine learning system designed to identify disaster-related and search-and-rescue (SAR) related social media posts.

The project focuses on automated classification of emergency-related tweets using classical machine learning and natural language processing (NLP) techniques. The framework includes preprocessing pipelines, feature extraction, machine learning model training, hyperparameter optimization, evaluation visualization, and interactive deployment using Streamlit and Docker.

The primary motivation of the project is to assist emergency response and disaster management systems by automatically filtering relevant emergency-related social media messages during large-scale disasters such as earthquakes.

---

# Project Objectives

The main objectives of the project are:

* Detect disaster-related tweets using machine learning
* Identify potential search-and-rescue related emergency messages
* Compare multiple machine learning algorithms
* Evaluate model performance using standard classification metrics
* Build an interactive web-based prediction interface
* Deploy the complete system using Docker

---

# Dataset

The project uses a disaster tweet dataset containing real disaster-related and non-disaster tweets.

## Dataset Statistics

| Property            | Value |
| ------------------- | ----- |
| Total Samples       | 7609  |
| Disaster Tweets     | 3271  |
| Non-disaster Tweets | 4338  |

The dataset undergoes preprocessing and text cleaning before feature extraction and model training.

---

# Machine Learning Pipeline

The project implements the following machine learning workflow:

```text
Raw Tweets
    ↓
Text Cleaning & Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
Machine Learning Classification
    ↓
Evaluation & Visualization
    ↓
Interactive Deployment
```

---

# Implemented Models

The following machine learning algorithms were implemented and evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Tuned SVM using GridSearchCV
* Multinomial Naive Bayes
* Random Forest
* Tuned Random Forest using GridSearchCV

---

# Best Model

The best performing model is the tuned Support Vector Machine (SVM).

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 79.63% |
| F1-score | 79.27% |

Best hyperparameters:

```python
{
    'svm__C': 0.1,
    'tfidf__max_features': 5000,
    'tfidf__ngram_range': (1, 2)
}
```

---

# Evaluation Metrics

The following evaluation metrics are used:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

The project additionally generates:

* confusion matrix visualizations,
* model comparison charts,
* and final result tables.

---

# Technologies Used

| Category             | Technologies  |
| -------------------- | ------------- |
| Programming Language | Python        |
| Machine Learning     | scikit-learn  |
| Data Processing      | pandas, numpy |
| Visualization        | matplotlib    |
| Web Interface        | Streamlit     |
| Containerization     | Docker        |
| Version Control      | Git & GitHub  |

---

# Project Structure

```text
DS570DSproject-SARsocialX/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   ├── best_svm_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest.pkl
│   └── best_random_forest.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison_f1.png
│   ├── confusion_matrix_tuned_svm.png
│   └── final_results_table.csv
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   └── evaluation/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/hamidmajidibalanji/DS570DSproject-SARsocialX.git

cd DS570DSproject-SARsocialX
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Data Preprocessing

Run preprocessing pipeline:

```bash
PYTHONPATH=. python src/preprocessing/preprocess.py
```

---

# Model Training

## Logistic Regression + SVM

```bash
PYTHONPATH=. python src/models/train_model.py
```

## Naive Bayes

```bash
PYTHONPATH=. python src/models/train_naive_bayes.py
```

## Random Forest

```bash
PYTHONPATH=. python src/models/train_random_forest.py
```

---

# Hyperparameter Tuning

## Tuned SVM

```bash
PYTHONPATH=. python src/models/grid_search_svm.py
```

## Tuned Random Forest

```bash
PYTHONPATH=. python src/models/grid_search_random_forest.py
```

---

# Evaluation & Visualization

Generate:

* confusion matrix,
* classification report,
* model comparison chart,
* final result table.

```bash
PYTHONPATH=. python src/evaluation/final_evaluation_figures.py
```

---

# Running the Streamlit Application

Run locally:

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit dashboard allows users to:

* enter custom disaster-related text,
* select machine learning models,
* perform predictions,
* visualize evaluation metrics,
* compare model performance.

---

# Docker Deployment

## Build Docker Image

```bash
docker build --no-cache -t sar-app .
```

## Run Docker Container

```bash
docker run -p 8501:8501 sar-app
```

Open:

```text
http://localhost:8501
```

---

# Experimental Results

## Model Performance Comparison

| Model               | Accuracy   | Precision  | Recall     | F1-score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.7957     | 0.7940     | 0.7957     | 0.7921     |
| SVM                 | 0.7582     | 0.7579     | 0.7582     | 0.7574     |
| Naive Bayes         | 0.7871     | 0.7840     | 0.7871     | 0.7830     |
| Random Forest       | 0.7825     | 0.7810     | 0.7825     | 0.7801     |
| Tuned SVM           | **0.7963** | **0.8000** | **0.7800** | **0.7927** |
| Tuned Random Forest | 0.7832     | 0.7820     | 0.7832     | 0.7818     |

---

# Future Work

Potential future extensions include:

* Transformer-based NLP models
* Real-time Twitter/X streaming
* Multi-language disaster tweet classification
* Geolocation-aware emergency prioritization
* Integration with autonomous SAR systems
* Real-time crisis monitoring dashboards

---

# Author

Hamid Majidi Balanji

Graduate Research Assistant
Networked and Autonomous Systems Laboratory (NAS Lab)
Özyeğin University
Istanbul, Türkiye

GitHub:
https://github.com/hamidmajidibalanji
