"""
train_emotion_models.py
======================
Multi-class emotion detection with Naive Bayes, Logistic Regression, and SVM.
Trains three models, computes metrics, and saves model artifacts.

Usage:
    python train_emotion_models.py

Output:
    - emotion_vectorizer.pkl
    - emotion_naive_bayes.pkl
    - emotion_logistic_regression.pkl
    - emotion_svm.pkl
    - metrics_summary.json (model comparison results)
"""

import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Download stopwords
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "emotions_dataset.csv"
VECTORIZER_PATH = ROOT_DIR / "emotion_vectorizer.pkl"
NB_MODEL_PATH = ROOT_DIR / "emotion_naive_bayes.pkl"
LR_MODEL_PATH = ROOT_DIR / "emotion_logistic_regression.pkl"
SVM_MODEL_PATH = ROOT_DIR / "emotion_svm.pkl"
METRICS_PATH = ROOT_DIR / "metrics_summary.json"

EMOTION_LABELS = ["happy", "sad", "angry", "fear", "neutral", "surprise"]


def preprocess(text: str) -> str:
    """Clean and normalize text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_dataset() -> Tuple[List[str], List[str]]:
    """Load and preprocess the emotions dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
    
    # Preprocess texts
    texts = [preprocess(t) for t in df["text"].tolist()]
    labels = df["emotion"].tolist()
    
    print(f"✓ Loaded {len(texts)} samples")
    print(f"✓ Emotion distribution: {dict(df['emotion'].value_counts())}")
    return texts, labels


def compute_metrics(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    train_time: float,
) -> Dict:
    """Compute comprehensive metrics for a model."""
    y_pred = model.predict(X_test)
    
    # Inference time
    start = time.perf_counter()
    for _ in range(100):
        model.predict(X_test[:1])
    inference_time = (time.perf_counter() - start) / 100 * 1000  # ms
    
    # Per-class metrics
    class_report = classification_report(
        y_test, y_pred, labels=EMOTION_LABELS, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=EMOTION_LABELS)
    
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "training_time_ms": round(train_time * 1000, 2),
        "inference_time_ms": round(inference_time, 2),
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "confusion_matrix": cm.tolist(),
        "class_report": {
            label: {
                "precision": round(class_report[label]["precision"], 4),
                "recall": round(class_report[label]["recall"], 4),
                "f1_score": round(class_report[label]["f1-score"], 4),
                "support": int(class_report[label]["support"]),
            }
            for label in EMOTION_LABELS
        },
    }
    
    return metrics


def train_models():
    """Train all three emotion classification models."""
    print("\n" + "=" * 70)
    print("EMOTION CLASSIFICATION MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    texts, labels = load_dataset()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n✓ Train/test split: {len(X_train)} / {len(X_test)}")
    
    # Vectorize
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✓ Vectorizer: {X_train_vec.shape[1]} features")
    
    # Save vectorizer
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Saved vectorizer to {VECTORIZER_PATH.name}")
    
    # Train models
    models_config = [
        ("Naive Bayes", MultinomialNB(alpha=1.0)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ("SVM (Linear)", SVC(kernel="linear", random_state=42, C=1.0)),
    ]
    
    metrics_list = []
    model_paths = {
        "Naive Bayes": NB_MODEL_PATH,
        "Logistic Regression": LR_MODEL_PATH,
        "SVM (Linear)": SVM_MODEL_PATH,
    }
    
    print("\n" + "-" * 70)
    for model_name, model in models_config:
        print(f"\nTraining {model_name}...")
        start = time.perf_counter()
        model.fit(X_train_vec, y_train)
        train_time = time.perf_counter() - start
        
        metrics = compute_metrics(
            model_name, model, X_train_vec, X_test_vec, y_train, y_test, train_time
        )
        metrics_list.append(metrics)
        
        # Save model
        with open(model_paths[model_name], "wb") as f:
            pickle.dump(model, f)
        
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ✓ Macro F1: {metrics['f1_macro']:.4f}")
        print(f"  ✓ Training time: {metrics['training_time_ms']:.1f} ms")
        print(f"  ✓ Inference time: {metrics['inference_time_ms']:.2f} ms/sample")
        print(f"  ✓ Saved to {model_paths[model_name].name}")
    
    # Save metrics summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(texts),
        "emotion_labels": EMOTION_LABELS,
        "train_test_split": f"{len(X_train)} / {len(X_test)}",
        "models": metrics_list,
        "best_model": max(metrics_list, key=lambda m: m["f1_macro"])["model"],
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Metrics summary saved to {METRICS_PATH.name}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 70)
    for m in metrics_list:
        print(
            f"{m['model']:<25} {m['accuracy']:<12.4f} {m['f1_macro']:<12.4f} {m['f1_weighted']:<12.4f}"
        )
    print("-" * 70)
    print(f"Best Model: {summary['best_model']}\n")


if __name__ == "__main__":
    train_models()
