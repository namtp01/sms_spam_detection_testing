---

### 💻 Methodology and Technology

This project was built with a clear separation of concerns, following industry best practices for a machine learning project.

#### 1. Machine Learning Model (Scikit-learn)

* **Vectorization (TF-IDF):** Messages are not raw text; they must be converted into numbers. I used `TfidfVectorizer` to transform each message into a numerical vector. This method down-weights common words (like "the", "a") and gives higher importance to words that are more unique to spam (like "prize", "free", "win").
* **Classifier (Multinomial Naive Bayes):** This is a classic, high-performance algorithm for text classification. It's extremely fast and effective for this task because it calculates the probability that a message is "spam" given the words it contains.

#### 2. Web Application (Streamlit)

* An interactive web UI was built using `Streamlit`. This allows the trained model to be used in real-time.
* The app loads the saved `model.pkl` and `tfidf.pkl` files, takes live user input, runs it through the *exact same* preprocessing pipeline, and serves the prediction.

#### 3. Testing Strategy (Pytest & Custom Scripts)

To ensure quality, the project uses two distinct forms of testing:

* **AI Model Testing (`test_model.py`):** This is an *adversarial* test script. It validates the model's intelligence and robustness by feeding it tricky messages (e.g., "fr33 prize", "I won our league") to see if it can be fooled.
* **Software Unit Testing (`test_project_logic.py`):** This uses the `pytest` framework to test the *application code*. It ensures that individual functions, like `preprocess_text`, work perfectly as expected (e.g., correctly removing punctuation, handling empty strings) and that the model files load without crashing.
