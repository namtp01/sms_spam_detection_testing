import pickle
import pytest
from app import preprocess_text, make_prediction  # Import our new function

# --- Test 1: Test the Preprocessing Function ---
# (These are the tests you already had)

def test_preprocess_lowercase():
    """Tests if the function correctly converts to lowercase."""
    assert preprocess_text("Hello World") == "hello world"

def test_preprocess_punctuation():
    """Tests if the function correctly removes punctuation."""
    assert preprocess_text("Hi! How are you?") == "hi how are you"

def test_preprocess_empty_string():
    """Tests if the function handles an empty string without crashing."""
    assert preprocess_text("") == ""

def test_preprocess_combined():
    """Tests a combined case."""
    assert preprocess_text("WINNER!! Click... NOW.") == "winner click now"

def test_preprocess_non_string():
    """Tests if the function handles non-string input gracefully."""
    assert preprocess_text(None) == ""
    assert preprocess_text(123) == ""

# --- Test 2: Test the Model Loading ---
# (This test you also had)

def test_model_files_load():
    """Tests if the .pkl files can be loaded without error."""
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
    except Exception as e:
        assert False, f"Failed to load model files: {e}"
    
    assert model is not None
    assert tfidf is not None

# --- Test 3: Test the Prediction Logic ---
# (These are the NEW tests for your app)

def test_prediction_ham():
    """Tests the full prediction logic on a known HAM message."""
    ham_message = "Hey, are you free for the call later? Grabbing lunch."
    assert make_prediction(ham_message) == "ham"

def test_prediction_spam():
    """Tests the full prediction logic on a known SPAM message."""
    spam_message = "WIN FREE MONEY CLICK HERE"
    assert make_prediction(spam_message) == "spam"

def test_prediction_tricky_spam():
    """Tests the full prediction logic on a tricky SPAM message."""
    tricky_spam = "you are a w i n n e r!! claim your fr33 prize"
    assert make_prediction(tricky_spam) == "spam"

def test_prediction_on_empty_input():
    """Tests how the prediction logic handles an empty string."""
    assert make_prediction("") == "ham"

def test_prediction_on_whitespace():
    """Tests how the prediction logic handles a string with only spaces."""
    assert make_prediction("   ") == "ham"
