import pickle

# 1. Load the saved model and vectorizer
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Run 'python train.py' first.")
    exit()

print("✅ Model and TF-IDF loaded.")

# 2. List of messages to test
test_messages = [
    # Obvious Spam
    "WIN FREE MONEY CLICK HERE",
    # Obvious Ham
    "Hey, are you free for the call later? Grabbing lunch.",
    # Tricky Spam
    "you are a w i n n e r!! claim your fr33 prize",
    # Tricky Ham (has spam words)
    "Hi, I won our fantasy league! The prize is just bragging rights lol."
]

# 3. Preprocess and predict
# We must apply the *same* preprocessing
processed_messages = [text.lower().replace(r'[^\w\s]', '') for text in test_messages]
messages_tfidf = tfidf.transform(processed_messages)
predictions = model.predict(messages_tfidf)

# 4. Show results
print("\n--- 🧪 Adversarial Test Results ---")
for i in range(len(test_messages)):
    print(f"\nMessage: {test_messages[i]}")
    print(f"Predicted: [{predictions[i].upper()}]")