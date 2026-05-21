import joblib  

MODEL_PATH = "models/logistic_regression.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_tweet(tweet):
    
    tweet_vector = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vector)[0]  
    
    return prediction