"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SENTIMENT ANALYSIS ML PIPELINE - BEGINNER FRIENDLY                 ║
║                                                                              ║
║ Complete workflow: CSV Upload → Preprocessing → Vectorization → Train → Eval ║
║                                                                              ║
║ Features:                                                                    ║
║  ✓ CSV file upload with validation                                          ║
║  ✓ Text preprocessing (lowercase, stopwords, tokenization)                  ║
║  ✓ TF-IDF vectorization for text representation                             ║
║  ✓ 3 trained models: Naive Bayes, Logistic Regression, SVM                  ║
║  ✓ Comprehensive evaluation (accuracy, precision, recall, F1, confusion)    ║
║  ✓ Model comparison and ranking by performance                              ║
║  ✓ CSV/JSON report generation and download                                  ║
║  ✓ Full error handling for edge cases                                       ║
║  ✓ CORS enabled for frontend integration                                    ║
║                                                                              ║
║ Usage:                                                                       ║
║  python3 app_ml_clean.py                                                    ║
║  Then upload CSV to: POST http://127.0.0.1:5001/api/ml/upload               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import csv
import io
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Data & ML Libraries
import nltk
import numpy as np
import pandas as pd

# Flask & CORS for API
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Scikit-learn: preprocessing, vectorization, models, metrics
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & SETUP
# ══════════════════════════════════════════════════════════════════════════════

# Download English stopwords (words like "the", "is", "and" that don't add sentiment)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# Setup folders
ROOT_DIR = Path(__file__).resolve().parent  # Backend folder
UPLOAD_FOLDER = ROOT_DIR / "uploads"        # Where CSV files are saved
UPLOAD_FOLDER.mkdir(exist_ok=True)          # Create folder if it doesn't exist

print("📁 Upload folder:", UPLOAD_FOLDER)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for ML training.
    
    Steps:
    1. Convert to lowercase → "HELLO" becomes "hello"
    2. Remove URLs/mentions → "http://example.com" → ""
    3. Remove special characters → "Hello!" → "Hello"
    4. Remove extra spaces → "hello   world" → "hello world"
    5. Tokenize & remove stopwords → ["hello", "world"]
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text as string
        
    Example:
        >>> preprocess_text("I LOVE this product!!!")
        'love product'
    """
    # Step 1: Lowercase
    text = str(text).lower()
    
    # Step 2: Remove URLs, @mentions, #hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    
    # Step 3: Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Step 4: Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Step 5: Tokenize (split by spaces) and remove stopwords
    # Only keep words longer than 2 characters
    tokens = [word for word in text.split() 
              if word not in STOP_WORDS and len(word) > 2]
    
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# ML PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SentimentMLPipeline:
    """
    Complete machine learning pipeline for sentiment analysis.
    
    Handles entire workflow:
    - Loading & validating CSV data
    - Text preprocessing
    - TF-IDF vectorization (convert text → numbers)
    - Train/test splitting with stratification
    - Training 3 models
    - Evaluating models with multiple metrics
    """
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.vectorizer = None          # TF-IDF vectorizer (created during training)
        self.models = {}                # Trained models: {name: model_object}
        self.metrics = {}               # Evaluation results: {name: metrics_dict}
        self.X_train = None             # Training features (vectorized text)
        self.X_test = None              # Test features
        self.y_train = None             # Training labels (sentiments)
        self.y_test = None              # Test labels
        self.labels = []                # Unique sentiment classes found in data
    
    def load_csv(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load and validate CSV file.
        
        Required CSV format:
            text,sentiment
            "I love this!",Positive
            "This is bad.",Negative
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            (texts, sentiments) - Lists of text and labels
            
        Raises:
            ValueError: If CSV invalid, missing columns, or empty
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate columns exist
            if "text" not in df.columns or "sentiment" not in df.columns:
                raise ValueError(
                    f"❌ CSV must have 'text' and 'sentiment' columns.\n"
                    f"   Found columns: {df.columns.tolist()}"
                )
            
            # Check minimum dataset size
            if len(df) < 10:
                raise ValueError(
                    f"❌ Dataset too small: {len(df)} rows. Need at least 10 rows."
                )
            
            # Remove rows with missing values
            original_count = len(df)
            df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)
            removed_count = original_count - len(df)
            
            if len(df) == 0:
                raise ValueError("❌ CSV has no valid rows after removing nulls")
            
            if removed_count > 0:
                print(f"   ⚠️  Removed {removed_count} rows with missing values")
            
            # Extract data
            texts = df["text"].tolist()
            sentiments = df["sentiment"].tolist()
            
            # Find unique classes
            self.labels = sorted(list(set(sentiments)))
            
            # Validate we have enough classes
            if len(self.labels) < 2:
                raise ValueError(
                    f"❌ Need at least 2 sentiment classes. Found only {len(self.labels)}"
                )
            
            return texts, sentiments
            
        except FileNotFoundError:
            raise ValueError(f"❌ File not found: {file_path}")
        except pd.errors.ParserError:
            raise ValueError("❌ Invalid CSV format. Ensure proper formatting.")
        except Exception as e:
            raise ValueError(f"❌ Error loading CSV: {str(e)}")
    
    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Clean all texts in a list.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [preprocess_text(text) for text in texts]
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Convert text to numeric features using TF-IDF.
        
        TF-IDF (Term Frequency-Inverse Document Frequency):
        - Converts words to numbers
        - High value = important unique word in document
        - Low value = common word across all documents
        
        Args:
            texts: List of text strings
            
        Returns:
            Matrix of shape (num_samples, num_features)
        """
        if self.vectorizer is None:
            # Create vectorizer (only on training data)
            self.vectorizer = TfidfVectorizer(
                max_features=5000,          # Keep top 5000 most frequent words
                ngram_range=(1, 2),         # Use 1-2 word combinations
                min_df=2,                   # Word must appear in at least 2 docs
                max_df=0.8,                 # Word can't appear in >80% of docs
            )
            return self.vectorizer.fit_transform(texts)
        else:
            # Transform validation/test data
            return self.vectorizer.transform(texts)
    
    def train_models(self, X_train, y_train) -> Dict:
        """
        Train 3 models on training data.
        
        Models:
        1. Naive Bayes - Fast probability-based classifier
        2. Logistic Regression - Linear classifier with regularization
        3. SVM - Support Vector Machine for complex patterns
        
        Args:
            X_train: Training features (vectorized text)
            y_train: Training labels (sentiments)
            
        Returns:
            Dict with training results
        """
        results = {}
        
        # Model configurations
        models_config = {
            "Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            ),
            "SVM": SVC(
                kernel="linear",
                random_state=42,
                C=1.0,
                probability=True    # Enable probability estimates
            ),
        }
        
        # Train each model
        for model_name, model in models_config.items():
            try:
                start_time = time.perf_counter()
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start_time
                
                # Store trained model
                self.models[model_name] = model
                results[model_name] = {
                    "trained": True,
                    "train_time": train_time
                }
                print(f"   ✓ {model_name} trained in {train_time:.3f}s")
                
            except Exception as e:
                results[model_name] = {
                    "trained": False,
                    "error": str(e)
                }
                print(f"   ✗ {model_name} failed: {str(e)}")
        
        return results
    
    def evaluate_model(self, model_name: str, X_test, y_test) -> Dict:
        """
        Evaluate trained model on test data.
        
        Metrics:
        - Accuracy: % correct predictions
        - Precision: Of predicted positives, how many were correct?
        - Recall: Of actual positives, how many did we find?
        - F1-Score: Harmonic mean of precision & recall
        - Confusion Matrix: True vs predicted for each class
        
        Args:
            model_name: Name of trained model to evaluate
            X_test: Test features
            y_test: True test labels
            
        Returns:
            Dict with all evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"❌ Model '{model_name}' not trained")
        
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Confusion matrix (rows=true, cols=predicted)
        cm = confusion_matrix(y_test, y_pred, labels=self.labels)
        
        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            labels=self.labels,
            output_dict=True,
            zero_division=0
        )
        
        per_class_metrics = {}
        for label in self.labels:
            per_class_metrics[label] = {
                "precision": round(class_report[label]["precision"], 4),
                "recall": round(class_report[label]["recall"], 4),
                "f1": round(class_report[label]["f1-score"], 4),
                "support": int(class_report[label]["support"]),
            }
        
        # Compile all metrics
        metrics = {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_metrics,
            "test_samples": len(y_test),
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def full_pipeline(self, file_path: str) -> Dict:
        """
        Execute complete ML pipeline.
        
        Steps:
        1. Load CSV file
        2. Preprocess all text
        3. Split into train/test with stratification
        4. Vectorize using TF-IDF
        5. Train 3 models
        6. Evaluate each model
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dict with success status, statistics, and results
        """
        try:
            print("\n" + "="*75)
            print("🚀 STARTING ML SENTIMENT ANALYSIS PIPELINE")
            print("="*75)
            
            # STEP 1: Load and validate CSV
            print("\n1️⃣ LOAD CSV FILE")
            texts, sentiments = self.load_csv(file_path)
            print(f"   ✓ Loaded {len(texts)} samples")
            print(f"   ✓ Classes found: {self.labels}")
            
            # STEP 2: Preprocess text
            print("\n2️⃣ PREPROCESS TEXT")
            texts_clean = self.preprocess(texts)
            print(f"   ✓ Cleaned {len(texts_clean)} texts")
            print(f"   Example: '{texts[0][:50]}...' → '{texts_clean[0]}'")
            
            # STEP 3: Split data into training and testing sets
            print("\n3️⃣ SPLIT DATA (Train/Test)")
            
            # Calculate optimal test_size for small datasets
            num_samples = len(texts_clean)
            num_classes = len(self.labels)
            
            # Ensure each class has enough samples in test set
            min_samples_per_class = 2
            max_test_samples = (num_samples // num_classes - min_samples_per_class) * num_classes
            
            if max_test_samples > 0:
                test_size = max(0.1, min(0.2, max_test_samples / num_samples))
            else:
                test_size = 0.1
            
            # Use stratified split to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                texts_clean,
                sentiments,
                test_size=test_size,
                random_state=42,           # Fixed seed for reproducibility
                stratify=sentiments        # Keep same class proportions in train/test
            )
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            print(f"   ✓ Training samples: {len(X_train)} ({(len(X_train)/num_samples)*100:.1f}%)")
            print(f"   ✓ Testing samples:  {len(X_test)} ({(len(X_test)/num_samples)*100:.1f}%)")
            
            # STEP 4: Vectorize text using TF-IDF
            print("\n4️⃣ VECTORIZE TEXT (TF-IDF)")
            X_train_vec = self.vectorize(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            print(f"   ✓ Feature matrix created")
            print(f"   ✓ Total features: {X_train_vec.shape[1]}")
            print(f"   ✓ Train shape: {X_train_vec.shape}")
            print(f"   ✓ Test shape:  {X_test_vec.shape}")
            
            # STEP 5: Train all models
            print("\n5️⃣ TRAIN MODELS")
            train_results = self.train_models(X_train_vec, y_train)
            
            # STEP 6: Evaluate all models
            print("\n6️⃣ EVALUATE MODELS")
            all_metrics = {}
            for model_name in self.models.keys():
                metrics = self.evaluate_model(model_name, X_test_vec, y_test)
                all_metrics[model_name] = metrics
                accuracy = metrics["accuracy"]
                print(f"   ✓ {model_name:25} Accuracy: {accuracy:.2%}")
            
            # Find best model
            best_model = max(all_metrics, key=lambda k: all_metrics[k]["accuracy"])
            best_accuracy = all_metrics[best_model]["accuracy"]
            
            print("\n" + "="*75)
            print(f"✅ PIPELINE COMPLETE - Best Model: {best_model} ({best_accuracy:.2%})")
            print("="*75 + "\n")
            
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
                "best_model": best_model,
            }
            
        except Exception as e:
            print(f"\n❌ PIPELINE ERROR: {str(e)}\n")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_single(self, text: str, model_name: str = None) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Text to predict
            model_name: Which model to use (default: first model)
            
        Returns:
            Prediction with confidence scores
        """
        if not self.models:
            raise ValueError("❌ No models trained yet. Upload CSV first.")
        
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"❌ Model '{model_name}' not found")
        
        # Preprocess and vectorize text
        model = self.models[model_name]
        text_clean = preprocess_text(text)
        X = self.vectorizer.transform([text_clean])
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Get probability scores
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
        """
        Generate downloadable report of results.
        
        Args:
            format: "csv" or "json"
            
        Returns:
            (file_path, file_name)
        """
        if not self.metrics:
            raise ValueError("❌ No metrics available. Train models first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(["Sentiment Analysis - Model Comparison Report"])
            writer.writerow([f"Generated: {datetime.now().isoformat()}"])
            writer.writerow([])
            
            # Metrics table
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
            best_model = max(self.metrics, key=lambda k: self.metrics[k]["accuracy"])
            writer.writerow(["Best Model:", best_model])
            
            # Save
            filename = f"sentiment_report_{timestamp}.csv"
            filepath = UPLOAD_FOLDER / filename
            with open(filepath, "w") as f:
                f.write(output.getvalue())
            
            return str(filepath), filename
        
        elif format == "json":
            filename = f"sentiment_report_{timestamp}.json"
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
            raise ValueError("❌ Format must be 'csv' or 'json'")


# ══════════════════════════════════════════════════════════════════════════════
# FLASK API SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend integration

# Global pipeline instance
pipeline = SentimentMLPipeline()


@app.route("/api/ml/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "✓ OK",
        "service": "ML Sentiment Analysis Pipeline",
        "models_trained": len(pipeline.models) > 0,
    })


@app.route("/api/ml/upload", methods=["POST"])
def upload_csv():
    """
    Upload CSV and train models.
    
    Request:
        POST /api/ml/upload
        Content-Type: multipart/form-data
        files: [CSV file]
    
    Response:
        {
            "success": true,
            "data_statistics": {...},
            "models": {...},
            "best_model": "Logistic Regression"
        }
    """
    try:
        # Validate file provided
        if "file" not in request.files:
            return jsonify({"error": "❌ No file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "❌ No file selected"}), 400
        
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "❌ Only CSV files supported"}), 400
        
        # Save uploaded file
        filepath = UPLOAD_FOLDER / file.filename
        file.save(filepath)
        
        print(f"\n📤 Received: {file.filename}")
        
        # Run pipeline
        result = pipeline.full_pipeline(str(filepath))
        
        if not result["success"]:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = f"❌ Server error: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500


@app.route("/api/ml/metrics", methods=["GET"])
def get_metrics():
    """Get all model metrics."""
    if not pipeline.metrics:
        return jsonify({
            "error": "❌ No models trained. Upload CSV first."
        }), 400
    
    return jsonify({
        "models": pipeline.metrics,
        "best_model": max(
            pipeline.metrics,
            key=lambda k: pipeline.metrics[k]["accuracy"]
        ),
        "summary": {
            "num_models": len(pipeline.metrics),
            "labels": pipeline.labels,
        }
    })


@app.route("/api/ml/predict", methods=["POST"])
def predict():
    """
    Predict sentiment for single text.
    
    Request:
        POST /api/ml/predict
        {"text": "I love this!", "model": "Logistic Regression"}
    """
    try:
        if not pipeline.models:
            return jsonify({
                "error": "❌ No models trained. Upload CSV first."
            }), 400
        
        data = request.get_json()
        text = data.get("text", "").strip()
        model_name = data.get("model")
        
        if not text:
            return jsonify({"error": "❌ Text required"}), 400
        
        result = pipeline.predict_single(text, model_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/report/<format>", methods=["GET"])
def download_report(format):
    """Download report as CSV or JSON."""
    try:
        filepath, filename = pipeline.generate_report(format)
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/status", methods=["GET"])
def status():
    """Get current pipeline status."""
    return jsonify({
        "trained": len(pipeline.models) > 0,
        "models_count": len(pipeline.models),
        "test_samples": len(pipeline.y_test) if pipeline.y_test else 0,
        "labels": pipeline.labels,
    })


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*75)
    print("🚀 STARTING ML SENTIMENT ANALYSIS API")
    print("="*75)
    print("\n📊 Available Endpoints:")
    print("   POST   /api/ml/upload    - Upload CSV and train models")
    print("   GET    /api/ml/health    - Health check")
    print("   GET    /api/ml/metrics   - Get model metrics")
    print("   POST   /api/ml/predict   - Predict sentiment for text")
    print("   GET    /api/ml/report/<fmt> - Download report (csv/json)")
    print("   GET    /api/ml/status    - Get pipeline status")
    print("\n🔗 Server: http://127.0.0.1:5001")
    print("   ⚠️  Debug mode: ON")
    print("="*75 + "\n")
    
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
        use_reloader=True,
        threaded=True
    )
