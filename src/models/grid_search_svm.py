# Implementation of grid search for SVM hyperparameters 
# Hamid Majidi Balanji

# importing necessary libraries
import os   
import joblib  
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer  

from sklearn.svm import LinearSVC  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score   

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.evaluation.save_metrics import save_metrics    


INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

BEST_MODEL_PATH = "models/best_svm_model.pkl"

def run_grid_search():
    
    print("Loading dataset...")
    
    df = pd.read_csv(INPUT_PATH)
    
    X = df["clean_text"]   
    y = df["label"]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    print("Building pipeline ...")
    
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(stop_words="english")
        ),
        (
            "svm",
            LinearSVC()
        )
    ])
    
    param_grid = {

        "tfidf__max_features": [
            1000,
            3000,
            5000
        ],

        "tfidf__ngram_range": [
            (1,1),
            (1,2)
        ],

        "svm__C": [
            0.1,
            1,
            10
        ]
    }
    
    print("Starting Grid Search ...")  
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring="f1"
    )
    
    grid_search.fit(X_train, y_train) 
    
    # --------------------------------------------------
    # Save Grid Search Results
    # --------------------------------------------------

    os.makedirs("outputs", exist_ok=True)

    results_df = pd.DataFrame(grid_search.cv_results_)

    results_df.to_csv(
        "outputs/grid_search_results.csv",
        index=False
    )

    print("\nGrid Search results saved to outputs/grid_search_results.csv")

    print("\nBest Parameters:")

    print(grid_search.best_params_)

    print("\nBest CV Score:")

    print(grid_search.best_score_) 
    
    print("\nBest Parameters:")  
    
    print(grid_search.best_params_)
    
    print("\nBest CV Score:")
    
    print(grid_search.best_score_)  
    
    best_model = grid_search.best_estimator_  
    
    predictions = best_model.predict(X_test) 
    
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )
    
    save_metrics(
        "Tuned SVM",
        accuracy,
        precision,
        recall,
        f1
    )
    
    print("\nFinal Test Results:")  
    
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )
    
    
    print(f"\nTest Accuracy: {accuracy:.4f}")   
    
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        BEST_MODEL_PATH
    )

    print(
        f"\nBest model saved to {BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    run_grid_search()

    
    

        
    
    
    