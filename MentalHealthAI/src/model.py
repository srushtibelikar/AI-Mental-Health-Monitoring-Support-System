import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_model():

    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    model_folder = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(model_folder, "mental_health_model.pkl")
    vectorizer_path = os.path.join(model_folder, "tfidf_vectorizer.pkl")

    # Check file existence
    if not os.path.exists(input_path):
        print("❌ Processed file not found!")
        print("Expected location:", input_path)
        return

    df = pd.read_csv(input_path)

    # Features and Labels
    X = df["cleaned_text"]
    y = df["label"]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(model_folder, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅ Model trained and saved successfully!")
    print("📁 Saved in:", model_folder)


if __name__ == "__main__":
    train_model()
