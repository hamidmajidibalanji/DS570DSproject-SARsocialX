import joblib

def predict_text(text, model_name):

    if model_name == "Tuned SVM":

        model = joblib.load(
            "models/best_svm_model.pkl"
        )

        prediction = model.predict(
            [text]
        )[0]
        
    elif model_name == "Naive Bayes":

        model = joblib.load(
            "models/naive_bayes_model.pkl"
        )

        vectorizer = joblib.load(
            "models/tfidf_vectorizer_nb.pkl"
        )

        tweet_vector = vectorizer.transform(
            [text]
        )

        prediction = model.predict(
            tweet_vector
        )[0]  
        
    elif model_name == "Tuned Random Forest":

        model = joblib.load(
            "models/best_random_forest.pkl"
        )

        prediction = model.predict(
            [text]
        )[0]      

    else:

        vectorizer = joblib.load(
            "models/tfidf_vectorizer.pkl"
        )

        tweet_vector = vectorizer.transform(
            [text]
        )

        if model_name == "SVM":

            model = joblib.load(
                "models/svm_model.pkl"
            )

        elif model_name == "Logistic Regression":

            model = joblib.load(
                "models/logistic_regression.pkl"
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        prediction = model.predict(
            tweet_vector
        )[0]

    return prediction