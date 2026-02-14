import joblib
import scipy
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from src.preprocessing import clean_text

model = joblib.load("model/model_rf.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
sia = SentimentIntensityAnalyzer()

def predict_mental_health(text):
    text_clean = clean_text(text)
    sentiment = sia.polarity_scores(text_clean)["compound"]

    X_text = tfidf.transform([text_clean])
    X = scipy.sparse.hstack([X_text, np.array(sentiment).reshape(1,1)])

    prediction = model.predict(X)
    return prediction[0]
