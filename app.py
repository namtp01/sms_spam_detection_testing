import streamlit as st
import pickle
import re

# --- 1. LOAD THE SAVED MODELS ---
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


# --- 2. LOGIC FUNCTIONS (NOW TESTABLE) ---

def preprocess_text(text):
    """Cleans text by lowercasing and removing punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def make_prediction(user_input):
    """
    Takes raw user input, preprocesses it, and returns the prediction.
    
    Returns:
        str: 'spam' or 'ham'
    """
    cleaned_input = preprocess_text(user_input)
    
    # Handle empty or whitespace-only input after cleaning
    if not cleaned_input.strip():
        # We can't predict on an empty string, so we'll classify it as ham
        return 'ham' 
        
    # Vectorize the cleaned input
    input_tfidf = tfidf.transform([cleaned_input])
    
    # Predict using the loaded model
    prediction = model.predict(input_tfidf)[0]  # [0] gets the first (and only) prediction
    return prediction


# --- 3. BUILD THE STREAMLIT APP INTERFACE ---

st.title("SMS Spam Detector 📨")
st.write(
    "Enter a message to check if it's spam or ham (not spam). "
    "This app uses a Naive Bayes model trained on a Kaggle dataset."
)

# Create a text area for user input
user_input = st.text_area("Enter your message here:")

# Create a button to submit
if st.button("Predict"):
    
    # Check if the input is *visually* empty
    if not user_input.strip():
        st.warning("Please enter a message to predict.")
    else:
        # Call our testable prediction function
        prediction = make_prediction(user_input)
        
        # Display the result
        st.subheader("Prediction:")
        
        if prediction == "spam":
            st.error("This message is SPAM 🚫")
        else:
            st.success("This message is HAM (Not Spam) ✅")
