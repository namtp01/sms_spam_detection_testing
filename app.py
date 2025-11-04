import streamlit as st
import pickle
import re

# --- 1. LOAD THE SAVED MODELS ---
# We load the "tfidf" vectorizer and "model"
# 'rb' means 'read binary', which is required for pickle files

try:
    with open('models/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'tfidf.pkl' file not found in 'models/' folder. Please run train.py first.")
    st.stop()

try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'model.pkl' file not found in 'models/' folder. Please run train.py first.")
    st.stop()


# --- 2. PREPROCESSING FUNCTION ---
# This MUST be the *exact same* preprocessing function from your training script
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# --- 3. BUILD THE STREAMLIT APP INTERFACE ---

# Set the title of the web app
st.title("SMS Spam Detector 📨")

# Add a description
st.write(
    "Enter a message to check if it's spam or ham (not spam). "
    "This app uses a Naive Bayes model trained on a Kaggle dataset."
)

# Create a text area for user input
user_input = st.text_area("Enter your message here:")

# Create a button to submit
if st.button("Predict"):
    
    # Check if the input is empty
    if not user_input:
        st.warning("Please enter a message to predict.")
    else:
        # --- 4. MAKE A PREDICTION ---
        
        # 1. Preprocess the user's input
        cleaned_input = preprocess_text(user_input)
        
        # 2. Vectorize the cleaned input (transform it into numbers)
        # Note: We use .transform() NOT .fit_transform()
        input_tfidf = tfidf.transform([cleaned_input])
        
        # 3. Predict using the loaded model
        prediction = model.predict(input_tfidf)[0]  # [0] gets the first (and only) prediction
        
        # 4. Display the result
        st.subheader("Prediction:")
        
        if prediction == "spam":
            st.error("This message is SPAM 🚫")
        else:
            st.success("This message is HAM (Not Spam) ✅")