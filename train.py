import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score # Import accuracy to see our score

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
    print("Vectorizing text with TF-IDF (including stop_words)...")
    
    # --- *** IMPROVEMENT HERE *** ---
    # We added stop_words='english'
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    # --- ************************* ---
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test) # Transform test data for accuracy check
    
    # 5. Train Model
    print("Training Naive Bayes model (with alpha=0.1)...")
    
    # --- *** IMPROVEMENT HERE *** ---
    # We added alpha=0.1
    model = MultinomialNB(alpha=0.1)
    # --- ************************* ---
    
    model.fit(X_train_tfidf, y_train)
    
    print("✅ Model trained successfully.")
    
    # (Optional) Check accuracy on the test set
    y_pred = model.predict(X_test_tfidf)
    print(f"Model Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")
    
    # 6. Save Model and Vectorizer
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    print("💾 New, improved model and TF-IDF saved to 'models/' folder.")

# This makes the script runnable
if __name__ == "__main__":
    train_model()
