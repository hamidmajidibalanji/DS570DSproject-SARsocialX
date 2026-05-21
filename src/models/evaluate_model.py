# Evaluate the evalutaion metrics of the model

import pandas as pd 
import joblib
import matplotlib.pyplot as plt

# load sklearn metrics   
from sklearn.model_selection import train_test_split   
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

MODEL_PATH = "models/logistic_regression.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def evaluate_model():
    
    print("Loading dataset...")
    
    df = pd.read_csv(INPUT_PATH)
    
    x = df["clean_text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    print("Loding model and vectorizer...")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    X_test_tfidf = vectorizer.transform(X_test)   
    
    predictions = model.predict(X_test_tfidf)   
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions) 
    
    print(f"Accuracy: {accuracy:.4f}\n")
    print(f"Precision: {precision:.4f}\n")
    print(f"Recall: {recall:.4f}\n")
    print(f"F1 Score: {f1:.4f}\n")
    
    # Confusion Matrix   
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    
    plt.savefig("outputs/confusion_matrix.png")
    
    print("Saved confusion matrix.")
    
    
if __name__ == "__main__":
    evaluate_model()
    
    
    
    