# Train Multinomial Naive Bayes model
# Hamid Majidi Balanji

import os  
import joblib  
import pandas as pd  

from sklearn.model_selection import train_test_split   

from sklearn.feature_extraction.text import TfidfVectorizer   

from src.evaluation.save_metrics import save_metrics

from sklearn.naive_bayes import MultinomialNB  

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score    
)

INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

MODEL_PATH = "models/naive_bayes_model.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer_nb.pkl"


def train_naive_bayes():
    
    print("Loading dataset ...")   
    
    df  = pd.read_csv(INPUT_PATH)   
    
    X = df["clean_text"]
    y = df["label"]   
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2,
        random_state=42
    )
    
    print("Creating TF-IDF features ...")  
    
    vectorizer = TfidfVectorizer(
                                 stop_words="english",
                                 max_features=5000
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)  
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes model ...")   
    
    model = MultinomialNB()  
    
    model.fit(
        X_train_tfidf,
        y_train 
    )
    
    predictions = model.predict(
        X_test_tfidf
    )
    
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    accuracy = accuracy_score(y_test, predictions)   
    
    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )
    
    recall = recall_score(y_test, predictions, average="weighted")   
    
    f1 = f1_score(y_test, predictions, average="weighted")  
     
    save_metrics(
        "Naive Bayes",
        accuracy,
        precision,
        recall,
        f1
    )
    
    print(
        f"\nAccuracy: {accuracy:.4f}"
    )
    
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )
    
    os.makedirs(
        "models",
        exist_ok=True
    )
    
    joblib.dump(
        model,
        MODEL_PATH
    )
    
    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )
    
    print(
        f"\nModel saved to {MODEL_PATH}"
    )
    
    print(
        f"Vectorizer saved to {VECTORIZER_PATH}"
    )


if __name__ == "__main__":
    train_naive_bayes()
    
      
    
    
    