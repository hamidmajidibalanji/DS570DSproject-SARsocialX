import pandas as pd  
import joblib  

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score   
from sklearn.pipeline import Pipeline


INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

MODEL_PATH = "models/logistic_regression.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"



def train_model():
    
    print("Loading processed dataset...")
    
    df = pd.read_csv(INPUT_PATH)
    
    x = df["clean_text"]
    y = df["label"]
    print("Splitting dataset into training and testing sets...")   
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  
    
    print("Creating TF-IDF features...")  
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) 
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")

    predictions = model.predict(X_test_tfidf)

    report = classification_report(y_test, predictions)

    print(report)

    print("Saving model...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()