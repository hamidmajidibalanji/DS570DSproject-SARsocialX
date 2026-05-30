import joblib

def predict_text(text, model_path):

    model = joblib.load(model_path)

    prediction = model.predict([text])

    return prediction[0]