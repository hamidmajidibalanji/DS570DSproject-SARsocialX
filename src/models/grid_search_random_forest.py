# Hyperparameter Tuning for Random Forest   
# Hamid Majidi Balanji

import os  
import joblib  
import numpy as np
import pandas as pd 

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier  

from sklearn.pipeline import Pipeline  

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.evaluation.save_metrics import save_metrics



INPUT_PATH = "data/processed/processed_disaster_tweets.csv"

BEST_MODEL_PATH = "models/best_random_forest.pkl"


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
    
    print("Building pipeline...")  
    
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english"
            )
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ])
    
    param_grid = {

        "tfidf__max_features": [
            1000,
            3000,
            5000
        ],

        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "rf__n_estimators": [
            100,
            200,
            300
        ],

        "rf__max_depth": [
            10,
            20,
            None
        ],

        "rf__min_samples_split": [
            2,
            5,
            10
        ]
    }
    
    print("Starting Grid Search...")  
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
        verbose=2
    )
    
    grid_search.fit(
        X_train,
        y_train
    )    
    
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV Score:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(
        X_test
    )
    
    
    
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
        "Tuned Random Forest",
        accuracy,
        precision,
        recall,
        f1
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        best_model,
        BEST_MODEL_PATH
    )

    print(
        f"\nBest model saved to {BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    run_grid_search()