import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def preprocess_text(text):
    """Cleans text by lowercasing and removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def train_model():
    """Trains the spam detection model and saves it."""
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/spam.csv', encoding='latin-1')
    except FileNotFoundError:
        print("Error: 'data/spam.csv' not found.")
        return

    # 2. Clean and Preprocess
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['message'] = df['message'].apply(preprocess_text)
    
    # 3. Split Data
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Vectorize
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # 5. Train Model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    print("✅ Model trained successfully.")
    
    # 6. Save Model and Vectorizer
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    print("💾 Model and TF-IDF vectorizer saved to 'models/' folder.")

# This makes the script runnable
if __name__ == "__main__":
    train_model()