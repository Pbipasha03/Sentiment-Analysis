"""
app.py
======
Flask API for Microtext Sentiment Analysis with multi-class emotion detection.
This backend exposes /api routes for the React frontend at http://localhost:5173.

HOW TO START:
    python app.py

ENDPOINTS:
    GET  /api/healthz
    GET  /api/models/metrics
    POST /api/sentiment/analyze
    POST /api/models/train
    POST /api/models/compare
    
    // EMOTION MODEL ENDPOINTS (new)
    GET  /api/emotion/metrics
    POST /api/emotion/analyze
    POST /api/emotion/compare
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

# EMOTION MODEL PATHS
EMOTION_VECTORIZER_PATH = ROOT_DIR / "emotion_vectorizer.pkl"
EMOTION_NB_PATH = ROOT_DIR / "emotion_naive_bayes.pkl"
EMOTION_LR_PATH = ROOT_DIR / "emotion_logistic_regression.pkl"
EMOTION_SVM_PATH = ROOT_DIR / "emotion_svm.pkl"
EMOTIONS_DATASET_PATH = ROOT_DIR / "emotions_dataset.csv"
EMOTION_METRICS_PATH = ROOT_DIR / "metrics_summary.json"
EMOTION_LABELS = ["happy", "sad", "angry", "fear", "neutral", "surprise"]


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


def load_emotion_models():
    """Load all emotion classification models."""
    emotion_models = {}
    if EMOTION_NB_PATH.exists():
        emotion_models["naive_bayes"] = load_pickle(EMOTION_NB_PATH)
    if EMOTION_LR_PATH.exists():
        emotion_models["logistic_regression"] = load_pickle(EMOTION_LR_PATH)
    if EMOTION_SVM_PATH.exists():
        emotion_models["svm"] = load_pickle(EMOTION_SVM_PATH)
    return emotion_models


def load_emotion_metrics():
    """Load pre-computed emotion model metrics."""
    if EMOTION_METRICS_PATH.exists():
        with open(EMOTION_METRICS_PATH, "r") as f:
            return json.load(f)
    return None


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


def get_emotion_model(model_name):
    """Get emotion model by name."""
    if model_name in emotion_models:
        return emotion_models[model_name]
    # Default to first available emotion model
    return next(iter(emotion_models.values())) if emotion_models else None


def normalize_emotion_probs(model, vector):
    """Normalize emotion model probabilities."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        classes = list(model.classes_)
        probs = {label: float(proba[classes.index(label)]) if label in classes else 0.0 for label in EMOTION_LABELS}
    else:
        pred = model.predict(vector)[0]
        probs = {label: 0.0 for label in EMOTION_LABELS}
        probs[pred] = 1.0
    total = sum(probs.values()) or 1.0
    return {label: round(probs.get(label, 0.0) / total, 4) for label in EMOTION_LABELS}


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
print("Loaded sentiment models:", list(models.keys()))

# Load emotion models and vectorizer
emotion_vectorizer = None
emotion_models = {}
emotion_metrics = None

if EMOTION_VECTORIZER_PATH.exists():
    emotion_vectorizer = load_pickle(EMOTION_VECTORIZER_PATH)
    emotion_models = load_emotion_models()
    emotion_metrics = load_emotion_metrics()
    print("Loaded emotion models:", list(emotion_models.keys()))
    if emotion_metrics:
        print(f"  Emotion dataset: {emotion_metrics.get('dataset_size')} samples")
        print(f"  Best emotion model: {emotion_metrics.get('best_model')}")
else:
    print("Warning: emotion model files not found. Emotion endpoints will be unavailable.")
    print("         Run `python train_emotion_models.py` to train emotion models.")

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


@app.route("/api/sentiment/analyze-batch", methods=["POST"])
def api_analyze_batch():
    """Analyze sentiment for batch of texts (from CSV upload).
    
    Request body:
    {
        "data": {
            "texts": ["text1", "text2", ...],
            "model": "naive_bayes" | "logistic_regression" | "svm"
        }
    }
    
    Response:
    {
        "results": [
            {"text": "...", "label": "positive"/"negative"/"neutral", "confidence": 0.95},
            ...
        ],
        "summary": {
            "total": 10,
            "positive": 6,
            "negative": 2,
            "neutral": 2
        },
        "processingTimeMs": 123
    }
    """
    try:
        start_time = time.time()
        data = request.get_json() or {}
        batch_data = data.get("data", {})
        texts = batch_data.get("texts", [])
        model_name = batch_data.get("model", "naive_bayes")
        
        # Validate input
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Request must include 'data.texts' as non-empty array"}), 400
        
        if not all(isinstance(t, str) and t.strip() for t in texts):
            return jsonify({"error": "All texts must be non-empty strings"}), 400
        
        # Check if models are trained
        if not vectorizer or not models:
            return jsonify({"error": "Models not trained. Please train models first."}), 400
        
        # Process each text
        results = []
        for text in texts:
            try:
                # Preprocess
                clean_text = preprocess(text)
                
                # Vectorize
                vector = vectorizer.transform([clean_text])
                
                # Get model
                model = get_model(model_name)
                if not model:
                    model = next(iter(models.values()))
                
                # Get predictions
                scores = normalize_probs(model, vector)
                label = max(scores, key=scores.get)
                confidence = float(scores[label])
                
                results.append({
                    "text": text[:100],  # Truncate for display
                    "label": label.lower(),  # "positive", "negative", "neutral"
                    "confidence": round(confidence, 2),
                    "scores": {k: round(v, 3) for k, v in scores.items()}
                })
            except Exception as e:
                # If one text fails, still process others but mark as error
                results.append({
                    "text": text[:100],
                    "label": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # Compute summary
        summary = {
            "total": len(results),
            "positive": sum(1 for r in results if r.get("label") == "positive"),
            "negative": sum(1 for r in results if r.get("label") == "negative"),
            "neutral": sum(1 for r in results if r.get("label") == "neutral"),
        }
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return jsonify({
            "results": results,
            "summary": summary,
            "processingTimeMs": processing_time_ms,
            "model": model_name
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500


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


@app.route("/api/models/compare", methods=["POST"])
def api_compare_models():
    data = request.get_json() or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request body must include a non-empty 'text' field."}), 400

    comparisons = [build_sentiment_result_one(text, model_name) for model_name in ["naive_bayes", "logistic_regression", "svm"]]
    labels = [c["label"] for c in comparisons]
    consensus, agreement = consensus_label(labels)

    return jsonify({
        "text": text,
        "comparisons": comparisons,
        "consensus": consensus,
        "agreement": round(agreement, 3),
    })


# ─── EMOTION MODEL ENDPOINTS ───────────────────────────────────────────────────

@app.route("/api/emotion/metrics", methods=["GET"])
def api_emotion_metrics():
    """Return emotion model metrics from training."""
    if not emotion_metrics:
        return jsonify({
            "trained": False,
            "metrics": [],
            "bestModel": None,
            "error": "Emotion models not trained. Run train_emotion_models.py"
        }), 400
    
    return jsonify({
        "trained": True,
        "dataset_size": emotion_metrics.get("dataset_size"),
        "emotion_labels": emotion_metrics.get("emotion_labels"),
        "models": emotion_metrics.get("models", []),
        "best_model": emotion_metrics.get("best_model"),
        "train_test_split": emotion_metrics.get("train_test_split"),
    })


@app.route("/api/emotion/analyze", methods=["POST"])
def api_emotion_analyze():
    """Classify text emotion using the best emotion model."""
    if not emotion_vectorizer or not emotion_models:
        return jsonify({"error": "Emotion models not available"}), 400
    
    data = request.get_json() or {}
    text = data.get("text", "")
    model_name = data.get("model")  # Optional, defaults to best model
    
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request body must include a non-empty 'text' field."}), 400
    
    # Default to best model if not specified or if model doesn't exist
    if not model_name or model_name not in emotion_models:
        model_name = emotion_metrics.get("best_model", "naive_bayes") if emotion_metrics else "naive_bayes"
    
    model = get_emotion_model(model_name)
    if not model:
        return jsonify({"error": f"Emotion model '{model_name}' not available"}), 400
    
    # Preprocess and predict
    clean_text = preprocess(text)
    vector = emotion_vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    scores = normalize_emotion_probs(model, vector)
    confidence = scores.get(prediction, 0.0)
    
    return jsonify({
        "originalText": text,
        "emotion": prediction,
        "confidence": round(confidence, 4),
        "scores": scores,
        "model": model_name,
        "processedText": clean_text,
    })


@app.route("/api/emotion/compare", methods=["POST"])
def api_emotion_compare():
    """Compare all emotion models on the same text."""
    if not emotion_vectorizer or not emotion_models:
        return jsonify({"error": "Emotion models not available"}), 400
    
    data = request.get_json() or {}
    text = data.get("text", "")
    
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request body must include a non-empty 'text' field."}), 400
    
    clean_text = preprocess(text)
    vector = emotion_vectorizer.transform([clean_text])
    
    comparisons = []
    predictions = []
    
    for model_name, model in emotion_models.items():
        prediction = model.predict(vector)[0]
        scores = normalize_emotion_probs(model, vector)
        confidence = scores.get(prediction, 0.0)
        
        comparisons.append({
            "model": model_name,
            "emotion": prediction,
            "confidence": round(confidence, 4),
            "scores": scores,
        })
        predictions.append(prediction)
    
    # Compute consensus and agreement
    emotion_counts = {}
    for e in predictions:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    
    consensus = max(emotion_counts, key=emotion_counts.get)
    agreement = round(max(emotion_counts.values()) / len(predictions), 4)
    
    return jsonify({
        "text": text,
        "comparisons": comparisons,
        "consensus": consensus,
        "agreement": agreement,
        "modelCount": len(emotion_models),
    })


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
