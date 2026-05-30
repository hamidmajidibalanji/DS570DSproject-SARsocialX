# Train Pipeline for Disaster Tweets Classification

import pandas as pd  
import joblib
import os  

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC   

from skleran.model_selection import GridSearchCV    

from sklearn.metrics import ( 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,   
    classification_report
)
   
from sklearn.pipeline import Pipeline




INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

MODELS_DIR = "models"

MODEL_PATH = "models/logistic_regression.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


os.makedirs(MODELS_DIR, exist_ok=True)   


def evaluate_model(model, X_test, y_test, model_name):
    
    predictions = model.predict(X_test)  
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions) 
    
    
    
    
    print("\n")
    print("=" * 50)  
    print(f"{model_name} RESULTS")  
    print("=" * 50)  
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }




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
    
    joblib.dump(
        vectorizer,
        os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    )
    
    results = []   
    
    # --------------------------------------------------
    #  Logistic Regression
    # --------------------------------------------------    
    

    print("Training Logistic Regression model...")

    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train_tfidf, y_train)
    
    joblib.dump(
        lr_model,
        os.path.join(MODELS_DIR, "logistic_regression.pkl")
    )
    
    lr_results = evaluate_model(
        lr_model, X_test_tfidf,
        y_test, 
        "Logistic Regression")
    
    
    results.append(lr_results)  
    
    # --------------------------------------------------
    #  Linear SVM
    # --------------------------------------------------  
    print("\nTraining Linear SVM model...")
    
    svm_model = LinearSVC()  
    
    svm_model.fit(X_train_tfidf, y_train)   
    
    joblib.dump(
        svm_model,
        os.path.join(MODELS_DIR, "svm_model.pkl")
    )
    
    
    svm_results = evaluate_model(
        svm_model, 
        X_test_tfidf,
        y_test, 
        "SVM"
    )
    
    results.append(svm_results)
    
    
    # --------------------------------------------------
    #  Save Comparison REsults   
    # --------------------------------------------------
    
    results_df = pd.DataFrame(results)  
    
    os.makedirs("outputs", exist_ok=True)   
    
    results_df.to_csv(
        "outputs/model_comparison.csv", 
        index=False
    )
    
    
    print("\nSaved model comparison results.")
    
    print("\nTraining complete. Models and vectorizer saved to disk.")  
    
    
   
if __name__ == "__main__":
    train_model()