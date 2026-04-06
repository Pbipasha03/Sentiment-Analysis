"""
train_model.py
==============
Trains Naive Bayes and Logistic Regression models for sentiment analysis.
Saves the best model and TF-IDF vectorizer as .pkl files.

HOW TO RUN:
    python train_model.py

DATASET FORMAT:
    A CSV file with two columns:
        - text  : the tweet / sentence
        - label : positive / negative / neutral
"""

import pandas as pd
import pickle
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords (only needed once)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


# ─── 1. TEXT PREPROCESSING ───────────────────────────────────────────────────

def preprocess(text):
    """Clean a raw text string."""
    text = str(text).lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove @mentions
    text = re.sub(r"#\w+", "", text)                  # remove #hashtags
    text = re.sub(r"[^a-z\s]", "", text)              # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ─── 2. LOAD DATASET ─────────────────────────────────────────────────────────

# Replace "dataset.csv" with your actual CSV filename.
# Your CSV must have columns: "text" and "label"
DATASET_PATH = "dataset.csv"

print(f"Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Keep only the columns we need
df = df[["text", "label"]].dropna()

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")


# ─── 3. PREPROCESS TEXTS ─────────────────────────────────────────────────────

print("Preprocessing texts...")
df["clean_text"] = df["text"].apply(preprocess)


# ─── 4. SPLIT INTO TRAIN / TEST ──────────────────────────────────────────────

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}\n")


# ─── 5. TF-IDF VECTORIZATION ─────────────────────────────────────────────────

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)


# ─── 6. TRAIN MODELS ─────────────────────────────────────────────────────────

models = {
    "Naive Bayes":          MultinomialNB(),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
}

best_model  = None
best_name   = ""
best_acc    = 0.0

print("=" * 55)
print(f"{'Model':<25} {'Accuracy':>10}")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc   = accuracy_score(y_test, preds)
    print(f"{name:<25} {acc:>10.4f}")

    if acc > best_acc:
        best_acc   = acc
        best_name  = name
        best_model = model

print("=" * 55)
print(f"\nBest model: {best_name}  (accuracy = {best_acc:.4f})\n")

# Detailed report for the best model
best_preds = best_model.predict(X_test_tfidf)
print("Classification Report for best model:")
print(classification_report(y_test, best_preds))


# ─── 7. SAVE MODEL AND VECTORIZER ────────────────────────────────────────────

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("naive_bayes.pkl", "wb") as f:
    pickle.dump(models["Naive Bayes"], f)

with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(models["Logistic Regression"], f)

print("Saved:  model.pkl")
print("Saved:  vectorizer.pkl")
print("Saved:  naive_bayes.pkl")
print("Saved:  logistic_regression.pkl")
print("\nTraining complete. You can now start app.py.")
