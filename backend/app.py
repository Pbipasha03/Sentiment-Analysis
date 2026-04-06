"""
app.py
======
Flask API for Microtext Sentiment Analysis.
This backend exposes /api routes for the React frontend at http://localhost:5173.

HOW TO START:
    python app.py

ENDPOINTS:
    GET  /api/healthz
    GET  /api/models/metrics
    POST /api/sentiment/analyze
    POST /api/models/train
    POST /predict
"""

import json
import pickle
import re
import time
from pathlib import Path

import nltk
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Download stopwords (only needed once)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model.pkl"
VECTORIZER_PATH = ROOT_DIR / "vectorizer.pkl"
NB_MODEL_PATH = ROOT_DIR / "naive_bayes.pkl"
LR_MODEL_PATH = ROOT_DIR / "logistic_regression.pkl"
DATASET_PATH = ROOT_DIR / "dataset.csv"
LABELS = ["negative", "neutral", "positive"]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def preprocess(text):
    """Clean a raw text string — same steps as training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models():
    models = {}
    if NB_MODEL_PATH.exists():
        models["naive_bayes"] = load_pickle(NB_MODEL_PATH)
    if LR_MODEL_PATH.exists():
        models["logistic_regression"] = load_pickle(LR_MODEL_PATH)
    if not models and MODEL_PATH.exists():
        models["best"] = load_pickle(MODEL_PATH)
    return models


def get_model(model_name):
    if model_name == "svm":
        model_name = "logistic_regression"
    if model_name in models:
        return models[model_name]
    return models.get("logistic_regression") or models.get("naive_bayes") or models.get("best")


def normalize_probs(model, vector):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        classes = list(model.classes_)
        probs = {label: float(proba[classes.index(label)]) if label in classes else 0.0 for label in LABELS}
    else:
        pred = model.predict(vector)[0]
        probs = {label: 0.0 for label in LABELS}
        probs[pred] = 1.0
    total = sum(probs.values()) or 1.0
    return {label: probs.get(label, 0.0) / total for label in LABELS}


def build_sentiment_result_one(text, model_name):
    clean = preprocess(text)
    vector = vectorizer.transform([clean])
    model = get_model(model_name)
    scores = normalize_probs(model, vector)
    label = max(scores, key=scores.get)
    confidence = float(scores[label])
    return {
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "model": model_name,
        "processedText": clean,
    }


def build_sentiment_response(text, model_name="naive_bayes"):
    result = build_sentiment_result_one(text, model_name)
    return {
        "originalText": text,
        "result": result,
        "allModels": [
            build_sentiment_result_one(text, name)
            for name in ["naive_bayes", "logistic_regression"]
            if get_model(name)
        ],
        "keywords": result["processedText"].split(),
        "processingTimeMs": 0,
    }


def compute_class_report(y_true, y_pred):
    raw_report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    class_report = {}
    for label, data in raw_report.items():
        if label in LABELS:
            class_report[label] = {
                "precision": float(data["precision"]),
                "recall": float(data["recall"]),
                "f1Score": float(data["f1-score"]),
                "support": int(data["support"]),
            }
    return class_report


def build_metrics(model_name, model, texts, labels):
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    matrix = confusion_matrix(labels, y_pred, labels=LABELS).tolist()
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(labels, y_pred)),
        "precision": float(precision_score(labels, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, y_pred, average="macro", zero_division=0)),
        "f1Score": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
        "confusionMatrix": {"matrix": matrix, "labels": LABELS},
        "classReport": compute_class_report(labels, y_pred),
        "trainingSamples": len(texts),
        "testSamples": len(texts),
        "trainingTimeMs": 0,
    }


def build_model_metrics():
    if dataset_df is None:
        return {"trained": False, "metrics": []}

    cleaned_texts = dataset_df["clean_text"].tolist()
    labels = dataset_df["label"].tolist()
    metrics = []
    for model_name in ["naive_bayes", "logistic_regression"]:
        model = get_model(model_name)
        if not model:
            continue
        metrics.append(build_metrics(model_name, model, cleaned_texts, labels))

    return {
        "trained": bool(metrics),
        "metrics": metrics,
        "bestModel": metrics[0]["model"] if metrics else None,
        "lastTrainedAt": None,
    }


def make_wordcloud(texts, sentiment_filter=None):
    counts = {}
    for text in texts:
        clean = preprocess(text)
        for word in clean.split():
            counts[word] = counts.get(word, 0) + 1
    words = [{"text": word, "value": count} for word, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    return {"words": words[:100]}


def consensus_label(labels):
    if not labels:
        return "neutral"
    counts = {label: labels.count(label) for label in set(labels)}
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    agreement = max(counts.values()) / len(labels)
    return best, agreement


# ─── APP SETUP ─────────────────────────────────────────────────────────────────

if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model.pkl and vectorizer.pkl must exist in the backend folder. Run `python train_model.py` first."
    )

vectorizer = load_pickle(VECTORIZER_PATH)
models = load_models()
print("Loaded models:", list(models.keys()))

if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess)
    dataset_df = df
else:
    dataset_df = None
    print("Warning: dataset.csv not found. Some endpoints may be disabled.")

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:4173",
                "http://127.0.0.1:4173",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        }
    },
    methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Sentiment API is running"})


@app.route("/api/healthz", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/models/metrics", methods=["GET"])
def api_model_metrics():
    return jsonify(build_model_metrics())


@app.route("/api/sentiment/analyze", methods=["POST"])
def api_analyze_sentiment():
    data = request.get_json() or {}
    text = data.get("text", "")
    model_name = data.get("model", "naive_bayes")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request body must include a non-empty 'text' field."}), 400

    return jsonify(build_sentiment_response(text, model_name))


@app.route("/api/models/train", methods=["POST"])
def api_train_models():
    data = request.get_json() or {}
    if data.get("useDefaultDataset", True):
        if dataset_df is None:
            return jsonify({"error": "Default dataset not available."}), 400
        texts = dataset_df["text"].tolist()
        labels = dataset_df["label"].tolist()
    else:
        texts = data.get("texts")
        labels = data.get("labels")

    if not texts or not labels or len(texts) != len(labels):
        return jsonify({"error": "Request body must include valid texts and labels arrays."}), 400

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    clean_texts = [preprocess(text) for text in texts]
    new_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = new_vectorizer.fit_transform(clean_texts)

    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    nb.fit(X, labels)
    lr.fit(X, labels)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(new_vectorizer, f)
    with open(NB_MODEL_PATH, "wb") as f:
        pickle.dump(nb, f)
    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    best_model = nb if accuracy_score(labels, nb.predict(X)) >= accuracy_score(labels, lr.predict(X)) else lr
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    global vectorizer, models
    vectorizer = new_vectorizer
    models = load_models()

    metrics = build_model_metrics()
    return jsonify({"success": True, "metrics": metrics["metrics"], "bestModel": metrics.get("bestModel"), "totalTimeMs": 0})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request body must include a non-empty 'text' field."}), 400

    clean = preprocess(text)
    model_instance = get_model("naive_bayes")
    prediction = model_instance.predict(vectorizer.transform([clean]))[0]
    return jsonify({"sentiment": prediction, "original_text": text, "clean_text": clean})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
