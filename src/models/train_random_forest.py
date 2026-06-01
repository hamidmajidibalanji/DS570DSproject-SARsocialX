import os  
import joblib
import pandas as pd  


from sklearn.model_selection import train_test_split  

from sklearn.ensemble import RandomForestClassifier  

from sklearn.feature_extraction.text import TfidfVectorizer   

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
    )

INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

MODEL_PATH = "models/random_forest.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer_rf.pkl"


def train_random_forest():
    
    df = pd.read_csv(INPUT_PATH)  
    
    X = df['clean_text']
    y = df['label']  
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    
    
    model.fit(X_train_tfidf, y_train)
    
    predictions = model.predict(X_test_tfidf)  
    
    
    print(classification_report(y_test, predictions))   
    
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")
    
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


if __name__ == "__main__":
    train_random_forest()