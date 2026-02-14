import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


# ---------------- TEXT CLEANING FUNCTION ---------------- #

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(words)


# ---------------- MAIN PREPROCESS FUNCTION ---------------- #

def preprocess():

    # Get project root directory dynamically
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "data", "sample_data.csv")
    output_folder = os.path.join(BASE_DIR, "data", "processed")
    output_path = os.path.join(output_folder, "processed_data.csv")

    # Check if file exists
    if not os.path.exists(input_path):
        print("❌ Error: sample_data.csv not found!")
        print("Expected location:", input_path)
        return

    df = pd.read_csv(input_path)

    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Sentiment score
    df["sentiment_score"] = df["cleaned_text"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    # Create processed folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("✅ Preprocessing completed successfully!")
    print("📁 Processed file saved at:", output_path)


if __name__ == "__main__":
    preprocess()
