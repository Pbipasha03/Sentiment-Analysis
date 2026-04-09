"""
app_ml_complete.py
==================
Complete ML Pipeline: CSV Upload → Train → Evaluate → Report

Features:
- CSV file upload with 'text' and 'sentiment' columns
- Automatic text preprocessing (stopwords, lowercase, tokenization)
- TF-IDF vectorization
- Train 3 models: Naive Bayes, Logistic Regression, SVM
- Model evaluation: accuracy, precision, recall, F1, confusion matrix
- Model comparison and ranking
- CSV/JSON report generation
- Complete error handling

Usage:
    python app_ml_complete.py
    Then upload CSV to POST /api/ml/upload
"""

import csv
import io
import json
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Setup
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = ROOT_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    text = str(text).lower()
    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize and remove stopwords
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class MLPipeline:
    """Complete ML pipeline: preprocess → vectorize → train → evaluate."""

    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels = []

    def load_csv(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Load CSV with 'text' and 'sentiment' columns."""
        try:
            df = pd.read_csv(file_path)

            # Validate columns
            if "text" not in df.columns or "sentiment" not in df.columns:
                raise ValueError(
                    f"CSV must have 'text' and 'sentiment' columns. Found: {df.columns.tolist()}"
                )

            # Remove null values
            df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

            if len(df) == 0:
                raise ValueError("CSV has no valid rows with text and sentiment")

            texts = df["text"].tolist()
            sentiments = df["sentiment"].tolist()

            # Store unique labels
            self.labels = sorted(list(set(sentiments)))

            return texts, sentiments

        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")

    def preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess all texts."""
        return [preprocess_text(t) for t in texts]

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Vectorize texts using TF-IDF."""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
            )
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)

    def train_models(self, X_train, y_train) -> Dict:
        """Train all 3 models."""
        results = {}

        # Model configurations
        model_configs = {
            "Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            ),
            "SVM": SVC(kernel="linear", random_state=42, C=1.0, probability=True),
        }

        for model_name, model in model_configs.items():
            try:
                start = time.perf_counter()
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start

                self.models[model_name] = model
                results[model_name] = {"trained": True, "train_time": train_time}
                print(f"✓ {model_name} trained in {train_time:.3f}s")
            except Exception as e:
                results[model_name] = {"trained": False, "error": str(e)}
                print(f"✗ {model_name} failed: {str(e)}")

        return results

    def evaluate_model(self, model_name: str, X_test, y_test) -> Dict:
        """Evaluate a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")

        model = self.models[model_name]
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.labels)

        # Classification report
        class_report = classification_report(
            y_test, y_pred, labels=self.labels, output_dict=True, zero_division=0
        )

        # Per-class metrics
        per_class = {}
        for label in self.labels:
            per_class[label] = {
                "precision": round(class_report[label]["precision"], 4),
                "recall": round(class_report[label]["recall"], 4),
                "f1": round(class_report[label]["f1-score"], 4),
                "support": int(class_report[label]["support"]),
            }

        metrics = {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "test_samples": len(y_test),
        }

        self.metrics[model_name] = metrics
        return metrics

    def full_pipeline(self, file_path: str) -> Dict:
        """Run complete pipeline: load → preprocess → vectorize → train → evaluate."""
        try:
            # 1. Load data
            print("\n" + "="*70)
            print("STARTING ML PIPELINE")
            print("="*70)
            
            print("\n1️⃣ Loading CSV...")
            texts, sentiments = self.load_csv(file_path)
            print(f"   ✓ Loaded {len(texts)} samples")
            print(f"   ✓ Classes: {self.labels}")

            # 2. Preprocess
            print("\n2️⃣ Preprocessing text...")
            texts_processed = self.preprocess(texts)
            print(f"   ✓ Cleaned {len(texts_processed)} texts")

            # 3. Split data
            print("\n3️⃣ Splitting data...")
            # Dynamically calculate test_size for small datasets
            num_samples = len(texts_processed)
            num_classes = len(self.labels)
            min_samples_per_class = 2
            max_test_samples = (num_samples // num_classes - min_samples_per_class) * num_classes
            test_size = max(0.1, min(0.2, max_test_samples / num_samples)) if max_test_samples > 0 else 0.1
            
            X_train, X_test, y_train, y_test = train_test_split(
                texts_processed, sentiments, test_size=test_size, random_state=42, stratify=sentiments
            )
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            print(f"   ✓ Train: {len(X_train)} samples")
            print(f"   ✓ Test: {len(X_test)} samples")

            # 4. Vectorize
            print("\n4️⃣ Vectorizing text (TF-IDF)...")
            X_train_vec = self.vectorize(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            print(f"   ✓ Features: {X_train_vec.shape[1]}")
            print(f"   ✓ Train shape: {X_train_vec.shape}")
            print(f"   ✓ Test shape: {X_test_vec.shape}")

            # 5. Train models
            print("\n5️⃣ Training models...")
            train_results = self.train_models(X_train_vec, y_train)

            # 6. Evaluate models
            print("\n6️⃣ Evaluating models...")
            all_metrics = {}
            for model_name in self.models.keys():
                metrics = self.evaluate_model(model_name, X_test_vec, y_test)
                all_metrics[model_name] = metrics
                print(f"   ✓ {model_name}: Accuracy={metrics['accuracy']:.2%}")

            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETE")
            print("="*70 + "\n")

            return {
                "success": True,
                "data_statistics": {
                    "total_samples": len(texts),
                    "classes": self.labels,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X_train_vec.shape[1],
                },
                "models": all_metrics,
                "best_model": max(all_metrics, key=lambda k: all_metrics[k]["accuracy"]),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_single(self, text: str, model_name: str = None) -> Dict:
        """Predict sentiment for a single text."""
        if not self.models:
            raise ValueError("No models trained yet")

        if model_name is None:
            model_name = list(self.models.keys())[0]

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        text_processed = preprocess_text(text)
        X = self.vectorizer.transform([text_processed])
        prediction = model.predict(X)[0]

        # Get probabilities if available
        scores = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            for label, prob in zip(model.classes_, proba):
                scores[label] = round(float(prob), 4)
        else:
            scores[prediction] = 1.0

        return {
            "text": text,
            "prediction": prediction,
            "scores": scores,
            "model": model_name,
            "confidence": round(max(scores.values()), 4),
        }

    def generate_report(self, format: str = "csv") -> Tuple[str, str]:
        """Generate downloadable report."""
        if not self.metrics:
            raise ValueError("No model metrics available")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow(["Sentiment Analysis Model Comparison Report"])
            writer.writerow([f"Generated: {datetime.now().isoformat()}"])
            writer.writerow([])

            # Summary table
            writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
            for model_name, metrics in self.metrics.items():
                writer.writerow([
                    model_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                ])

            writer.writerow([])
            writer.writerow(["Best Model:", max(self.metrics, key=lambda k: self.metrics[k]["accuracy"])])

            # Save to file
            filename = f"sentiment_analysis_report_{timestamp}.csv"
            filepath = UPLOAD_FOLDER / filename
            with open(filepath, "w") as f:
                f.write(output.getvalue())

            return str(filepath), filename

        elif format == "json":
            filename = f"sentiment_analysis_report_{timestamp}.json"
            filepath = UPLOAD_FOLDER / filename

            report = {
                "generated_at": datetime.now().isoformat(),
                "metrics": self.metrics,
                "best_model": max(self.metrics, key=lambda k: self.metrics[k]["accuracy"]),
            }

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            return str(filepath), filename

        else:
            raise ValueError("Format must be 'csv' or 'json'")


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

# Global pipeline instance
pipeline = MLPipeline()


@app.route("/api/ml/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "service": "ML Sentiment Analysis Pipeline",
        "models_trained": len(pipeline.models) > 0,
    })


@app.route("/api/ml/upload", methods=["POST"])
def upload_csv():
    """Upload CSV file and train models."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        # Save uploaded file
        filepath = UPLOAD_FOLDER / file.filename
        file.save(filepath)

        # Run pipeline
        result = pipeline.full_pipeline(str(filepath))

        if not result["success"]:
            return jsonify({"error": result["error"]}), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/ml/metrics", methods=["GET"])
def get_metrics():
    """Get all model evaluation metrics."""
    if not pipeline.metrics:
        return jsonify({"error": "No models trained yet. Please upload CSV first."}), 400

    return jsonify({
        "models": pipeline.metrics,
        "best_model": max(pipeline.metrics, key=lambda k: pipeline.metrics[k]["accuracy"]),
        "summary": {
            "num_models": len(pipeline.metrics),
            "labels": pipeline.labels,
        }
    })


@app.route("/api/ml/predict", methods=["POST"])
def predict():
    """Predict sentiment for single text."""
    try:
        if not pipeline.models:
            return jsonify({"error": "No models trained. Please upload CSV first."}), 400

        data = request.get_json()
        text = data.get("text")
        model_name = data.get("model")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        result = pipeline.predict_single(text, model_name)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/report/<format>", methods=["GET"])
def download_report(format):
    """Download report as CSV or JSON."""
    try:
        if not pipeline.metrics:
            return jsonify({"error": "No metrics available. Train models first."}), 400

        filepath, filename = pipeline.generate_report(format)

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype="text/csv" if format == "csv" else "application/json",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/status", methods=["GET"])
def get_status():
    """Get pipeline status."""
    return jsonify({
        "trained": len(pipeline.models) > 0,
        "num_models": len(pipeline.models),
        "num_metrics": len(pipeline.metrics),
        "labels": pipeline.labels,
        "best_model": max(pipeline.metrics, key=lambda k: pipeline.metrics[k]["accuracy"]) if pipeline.metrics else None,
    })


if __name__ == "__main__":
    print("🚀 Starting ML Sentiment Analysis API...")
    print("📊 Endpoints:")
    print("   POST /api/ml/upload       - Upload CSV and train models")
    print("   GET  /api/ml/metrics      - Get model evaluation metrics")
    print("   POST /api/ml/predict      - Predict sentiment for text")
    print("   GET  /api/ml/report/<fmt> - Download report (csv/json)")
    print("   GET  /api/ml/status       - Get pipeline status")
    print("\n")
    app.run(host="127.0.0.1", port=5001, debug=True)
