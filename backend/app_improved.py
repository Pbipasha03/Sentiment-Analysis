"""
app_improved.py
===============
Improved Flask API for Sentiment Analysis with:
- Better text preprocessing (negation-aware)
- N-gram feature engineering (captures phrases)
- Balanced dataset and class weighting
- Confidence threshold-based neutral classification
- Rule-based neutral detection
- Comprehensive evaluation metrics

HOW TO START:
    python app_improved.py

ENDPOINTS:
    GET  /api/healthz
    POST /api/sentiment/analyze
    POST /api/sentiment/analyze-batch
    GET  /api/models/metrics
"""

import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, Tuple

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

# Download NLTK resources
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ─── CONFIGURATION ─────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = ROOT_DIR / "vectorizer_improved.pkl"
LR_MODEL_PATH = ROOT_DIR / "sentiment_lr_improved.pkl"
DATASET_PATH = ROOT_DIR / "dataset.csv"

LABELS = ["negative", "neutral", "positive"]
CONFIDENCE_THRESHOLD = 0.4  # Below this → classify as neutral


# ─── IMPROVED PREPROCESSING ────────────────────────────────────────────────

def get_improved_stopwords():
    """Get stopwords but KEEP negation words like 'not', 'neither', 'nor'."""
    stop_words = set(stopwords.words("english"))
    keep_words = {
        "not", "no", "nor", "neither", "never", "nothing", "nowhere",
        "nobody", "isn", "aren", "wasn", "weren", "hasn", "hadn",
        "haven", "doesnt", "didnt", "dont", "should", "shouldnt",
        "wont", "wouldnt", "cant", "couldnt", "shan", "shant"
    }
    return stop_words - keep_words


STOP_WORDS_IMPROVED = get_improved_stopwords()


def preprocess_improved(text: str) -> str:
    """
    Enhanced preprocessing preserving negation and important phrases.
    
    Preserves: "not good", "neither good nor bad", "isn't bad"
    Removes: Common stopwords except negations
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS_IMPROVED and len(w) > 2]
    return " ".join(tokens)


def rule_based_neutral_detection(text: str) -> Tuple[bool, float]:
    """
    Rule-based detection for explicit neutral phrases.
    Examples: "neither good nor bad", "average", "okay"
    Returns: (is_neutral, confidence)
    """
    text_lower = text.lower()
    neutral_phrases = [
        r"neither.*nor", r"neither good nor bad", r"average", r"okay",
        r"so so", r"not bad not good", r"just okay", r"moderate",
        r"alright", r"fine", r"decent", r"nothing special"
    ]
    
    for phrase in neutral_phrases:
        if re.search(phrase, text_lower):
            return True, 0.95
    
    return False, 0.0


# ─── MODEL LOADING ────────────────────────────────────────────────────────

def load_models():
    """Load vectorizer and model."""
    if not VECTORIZER_PATH.exists() or not LR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model files not found. Run train_improved_models.py first.\n"
            f"  Expected: {VECTORIZER_PATH}\n"
            f"  Expected: {LR_MODEL_PATH}"
        )
    
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    
    with open(LR_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    return vectorizer, model


# ─── PREDICTION LOGIC ─────────────────────────────────────────────────────

def predict_sentiment(text: str, vectorizer, model, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Dict:
    """
    Predict sentiment with improved logic.
    
    1. Check rule-based neutral phrases first
    2. If low confidence (< threshold) → classify as neutral
    3. Return probabilities for all classes
    
    Returns:
        {
            "label": str,
            "confidence": float,
            "probabilities": dict,
            "method": str ("rule", "threshold", "model"),
            "cleaned_text": str
        }
    """
    # Rule-based neutral detection
    is_neutral_rule, conf_rule = rule_based_neutral_detection(text)
    
    # Preprocess
    clean_text = preprocess_improved(text)
    
    if not clean_text:
        return {
            "label": "neutral",
            "confidence": 0.5,
            "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            "method": "empty_text",
            "cleaned_text": clean_text
        }
    
    # Vectorize
    X = vectorizer.transform([clean_text])
    
    # Get prediction
    pred_label = model.predict(X)[0]
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        label_idx = list(model.classes_).index(pred_label)
        confidence = float(proba[label_idx])
        
        # Create probability dict with proper string keys (not numpy strings)
        probabilities = {}
        for i, cls_label in enumerate(model.classes_):
            probabilities[str(cls_label)] = float(proba[i])
    else:
        confidence = 1.0
        probabilities = {LABELS[0]: 0.0, pred_label: 1.0, LABELS[-1]: 0.0}
    
    # Apply decision logic
    
    # 1. Rule-based neutral detection (highest priority)
    if is_neutral_rule:
        return {
            "label": "neutral",
            "confidence": conf_rule,
            "probabilities": probabilities,
            "method": "rule_based",
            "cleaned_text": clean_text
        }
    
    # 2. Confidence threshold (convert low confidence to neutral)
    if confidence < confidence_threshold:
        neutral_confidence = 1.0 - (confidence / confidence_threshold)
        return {
            "label": "neutral",
            "confidence": round(neutral_confidence, 3),
            "probabilities": probabilities,
            "method": "confidence_threshold",
            "cleaned_text": clean_text,
            "original_label": pred_label,
            "original_confidence": round(confidence, 3)
        }
    
    # 3. Normal model prediction
    return {
        "label": pred_label,
        "confidence": round(confidence, 3),
        "probabilities": {k: round(v, 3) for k, v in probabilities.items()},
        "method": "model",
        "cleaned_text": clean_text
    }


# ─── FLASK APP ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

try:
    vectorizer, model = load_models()
    print("\n✅ Models loaded successfully")
    print(f"   Vectorizer features: {vectorizer.get_feature_names_out().shape[0]}")
    print(f"   Model classes: {list(model.classes_)}")
except FileNotFoundError as e:
    print(f"\n❌ {e}")
    exit(1)


@app.route("/api/healthz", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Improved sentiment API running"}), 200


@app.route("/api/sentiment/analyze", methods=["POST"])
def api_analyze():
    """
    Analyze sentiment for a single text.
    
    Request body:
    {
        "text": "I love this product!",
        "model": "logistic_regression" (optional, ignored in improved version)
    }
    
    Response:
    {
        "text": "I love this product!",
        "result": {
            "label": "positive",
            "confidence": 0.85,
            "probabilities": {"negative": 0.05, "neutral": 0.1, "positive": 0.85},
            "method": "model",
            "cleaned_text": "love product"
        },
        "processingTimeMs": 12
    }
    """
    try:
        start_time = time.time()
        
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Request body must include non-empty 'text' field"}), 400
        
        # Predict
        result = predict_sentiment(text, vectorizer, model, CONFIDENCE_THRESHOLD)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return jsonify({
            "text": text,
            "result": result,
            "processingTimeMs": processing_time_ms
        }), 200
    
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/sentiment/analyze-batch", methods=["POST"])
def api_analyze_batch():
    """
    Analyze sentiment for batch of texts (from CSV upload).
    
    Request body:
    {
        "texts": ["text1", "text2", ...],
        "model": "logistic_regression" (optional)
    }
    """
    try:
        start_time = time.time()
        
        json_data = request.get_json() or {}
        
        # Extract texts - support both flat and wrapped formats
        if 'texts' in json_data:
            texts = json_data.get('texts', [])
        elif 'data' in json_data:
            texts = json_data.get('data', {}).get('texts', [])
        else:
            texts = []
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Request must include 'texts' as non-empty array"}), 400
        
        # Remove empty strings
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        
        if not texts:
            return jsonify({"error": "All texts were empty"}), 400
        
        # Process each text
        results = []
        errors = 0
        
        for idx, text in enumerate(texts):
            try:
                pred = predict_sentiment(text, vectorizer, model, CONFIDENCE_THRESHOLD)
                results.append({
                    "index": idx,
                    "text": text[:100],  # Truncate for display
                    "label": pred["label"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "method": pred.get("method", "model")
                })
            except Exception as e:
                errors += 1
                results.append({
                    "index": idx,
                    "text": text[:100],
                    "label": "neutral",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # Summary statistics
        label_counts = {}
        for r in results:
            label = r.get("label", "neutral")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        summary = {
            "total": len(results),
            "negative": label_counts.get("negative", 0),
            "neutral": label_counts.get("neutral", 0),
            "positive": label_counts.get("positive", 0),
            "errors": errors
        }
        
        # Calculate percentages
        total = summary["total"]
        if total > 0:
            summary["negativePercent"] = round((summary["negative"] / total) * 100, 1)
            summary["neutralPercent"] = round((summary["neutral"] / total) * 100, 1)
            summary["positivePercent"] = round((summary["positive"] / total) * 100, 1)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        print(f"✅ Batch processed: {total} texts, {errors} errors, {processing_time_ms}ms")
        
        return jsonify({
            "results": results,
            "summary": summary,
            "processingTimeMs": processing_time_ms
        }), 200
    
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500


@app.route("/api/models/metrics", methods=["GET"])
def api_metrics():
    """
    Return model evaluation metrics on the training dataset.
    Shows confusion matrix and per-class metrics, especially for Neutral.
    """
    try:
        if not DATASET_PATH.exists():
            return jsonify({
                "trained": False,
                "error": "Dataset not found for metrics calculation"
            }), 400
        
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        if "label" not in df.columns:
            return jsonify({"trained": False, "error": "Dataset missing 'label' column"}), 400
        
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        # Preprocess and vectorize
        clean_texts = [preprocess_improved(t) for t in texts]
        X = vectorizer.transform(clean_texts)
        
        # Predict
        y_pred = model.predict(X)
        
        # Metrics
        accuracy = accuracy_score(labels, y_pred)
        precision = precision_score(labels, y_pred, average="macro", zero_division=0)
        recall = recall_score(labels, y_pred, average="macro", zero_division=0)
        f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, y_pred, labels=LABELS)
        
        # Per-class report (IMPORTANT for neutral)
        report = classification_report(labels, y_pred, labels=LABELS, output_dict=True, zero_division=0)
        
        class_metrics = {}
        for label in LABELS:
            class_metrics[label] = {
                "precision": round(report[label]["precision"], 3),
                "recall": round(report[label]["recall"], 3),
                "f1_score": round(report[label]["f1-score"], 3),
                "support": int(report[label]["support"])
            }
        
        return jsonify({
            "trained": True,
            "model": "Logistic Regression (Improved)",
            "preprocessing": "improved (negation-aware n-grams)",
            "dataset_samples": len(texts),
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_score_macro": round(f1, 4),
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": LABELS
            },
            "per_class_metrics": class_metrics,
            "notes": {
                "neutral": "Neutral class uses confidence threshold + rule-based detection",
                "preprocessing": "Keeps negation words: not, neither, nor, never",
                "features": "TF-IDF with bigrams (n-grams=1,2) for phrase capture",
                "confidence_threshold": CONFIDENCE_THRESHOLD
            }
        }), 200
    
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = 5002  # Use port 5002 to avoid macOS AirPlay conflict
    print("\n" + "="*70)
    print("🚀 IMPROVED SENTIMENT API")
    print("="*70)
    print(f"📍 Running on http://127.0.0.1:{port}")
    print(f"✨ Features:")
    print(f"   - Negation-aware preprocessing (keeps 'not', 'neither', 'nor')")
    print(f"   - N-grams for phrase understanding")
    print(f"   - Confidence threshold ({CONFIDENCE_THRESHOLD}) for neutral classification")
    print(f"   - Rule-based neutral detection")
    print(f"   - Balanced dataset with class weighting")
    print(f"\n📊 Endpoints:")
    print(f"   GET  /api/healthz")
    print(f"   POST /api/sentiment/analyze")
    print(f"   POST /api/sentiment/analyze-batch")
    print(f"   GET  /api/models/metrics")
    print("="*70 + "\n")
    
    app.run(host="127.0.0.1", port=port, debug=True)
