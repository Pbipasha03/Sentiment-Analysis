"""
streamlit_ml_app.py
===================
Complete Sentiment Analysis ML Pipeline with Streamlit
- CSV Upload with validation
- Text preprocessing (lowercase, stopwords, tokenization)
- TF-IDF vectorization
- Train 3 models: Naive Bayes, Logistic Regression, SVM
- Display metrics, confusion matrix, per-class scores
- Test predictions on new text
- Download reports (CSV/JSON)

Run: streamlit run streamlit_ml_app.py
"""

import io
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
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

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

STOP_WORDS = download_nltk_data()

# ═════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════
# ML PIPELINE CLASS
# ═════════════════════════════════════════════════════════════════════════

class MLPipeline:
    """Complete ML pipeline: preprocess → vectorize → train → evaluate."""

    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.metrics = {}
        self.labels = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_csv(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Load and validate DataFrame."""
        if "text" not in df.columns or "sentiment" not in df.columns:
            raise ValueError(
                f"CSV must have 'text' and 'sentiment' columns. Found: {df.columns.tolist()}"
            )

        df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No valid rows with text and sentiment")

        texts = df["text"].tolist()
        sentiments = df["sentiment"].tolist()
        self.labels = sorted(list(set(sentiments)))

        return texts, sentiments

    def preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess all texts."""
        return [preprocess_text(t) for t in texts]

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Vectorize using TF-IDF."""
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
        model_configs = {
            "Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            ),
            "SVM": SVC(kernel="linear", random_state=42, C=1.0, probability=True),
        }

        for model_name, model in model_configs.items():
            try:
                model.fit(X_train, y_train)
                self.models[model_name] = model
                results[model_name] = {"trained": True}
            except Exception as e:
                results[model_name] = {"trained": False, "error": str(e)}

        return results

    def evaluate_model(self, model_name: str, X_test, y_test) -> Dict:
        """Evaluate a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")

        model = self.models[model_name]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.labels)
        class_report = classification_report(
            y_test, y_pred, labels=self.labels, output_dict=True, zero_division=0
        )

        per_class = {}
        for label in self.labels:
            per_class[label] = {
                "precision": round(class_report[label]["precision"], 4),
                "recall": round(class_report[label]["recall"], 4),
                "f1": round(class_report[label]["f1-score"], 4),
                "support": int(class_report[label]["support"]),
            }

        return {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "test_samples": len(y_test),
        }

    def full_pipeline(self, df: pd.DataFrame) -> Dict:
        """Run complete pipeline."""
        try:
            # Load and validate
            texts, sentiments = self.load_csv(df)

            # Preprocess
            texts_processed = self.preprocess(texts)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts_processed,
                sentiments,
                test_size=0.2,
                random_state=42,
                stratify=sentiments,
            )
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            # Vectorize
            X_train_vec = self.vectorize(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            # Train models
            self.train_models(X_train_vec, y_train)

            # Evaluate all models
            all_metrics = {}
            for model_name in self.models.keys():
                metrics = self.evaluate_model(model_name, X_test_vec, y_test)
                all_metrics[model_name] = metrics

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
                "best_model": max(
                    all_metrics, key=lambda k: all_metrics[k]["accuracy"]
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_single(self, text: str, model_name: str = None) -> Dict:
        """Predict sentiment for single text."""
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

        # Get confidence scores
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)[0]
        else:
            # For SVM without probability
            probas = None

        scores = {}
        if probas is not None:
            for i, label in enumerate(self.labels):
                scores[label] = float(probas[i])
        else:
            scores = {label: 0.0 for label in self.labels}
            scores[prediction] = 1.0

        return {
            "prediction": prediction,
            "confidence": scores.get(prediction, 1.0),
            "scores": scores,
            "text": text,
        }

    def generate_report(self, format: str = "csv") -> str:
        """Generate CSV or JSON report."""
        if not self.metrics:
            return ""

        if format == "csv":
            output = io.StringIO()
            output.write("Sentiment Analysis Model Comparison Report\n")
            output.write(f"Generated: {datetime.now().isoformat()}\n\n")
            output.write("Model,Accuracy,Precision,Recall,F1-Score\n")

            for model_name, metrics in self.metrics.items():
                output.write(
                    f"{model_name},{metrics['accuracy']},{metrics['precision']},"
                    f"{metrics['recall']},{metrics['f1']}\n"
                )

            return output.getvalue()

        elif format == "json":
            return json.dumps(self.metrics, indent=2)

        return ""


# ═════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Sentiment Analysis ML Pipeline",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Sentiment Analysis ML Pipeline")
    st.markdown(
        "Upload CSV, train models, and analyze sentiment with Naive Bayes, Logistic Regression & SVM"
    )

    # Initialize session state
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = MLPipeline()
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "metrics" not in st.session_state:
        st.session_state.metrics = None

    # ─── SECTION 1: CSV UPLOAD & TRAINING ─────────────────────────────────

    st.header("📁 Step 1: Upload & Train")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file (must have 'text' and 'sentiment' columns)",
            type="csv",
        )

    with col2:
        if uploaded_file:
            df_preview = pd.read_csv(uploaded_file)
            st.success(f"✓ File loaded: {len(df_preview)} rows")

    # Train button
    if st.button("🚀 Train Models", use_container_width=True, type="primary"):
        if not uploaded_file:
            st.error("❌ Please upload a CSV file first")
        else:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)

                # Validate
                if "text" not in df.columns or "sentiment" not in df.columns:
                    st.error(
                        f"❌ CSV must have 'text' and 'sentiment' columns. Found: {df.columns.tolist()}"
                    )
                else:
                    with st.spinner("⏳ Training models... this may take a minute"):
                        # Run pipeline
                        result = st.session_state.pipeline.full_pipeline(df)

                        if result["success"]:
                            st.session_state.trained = True
                            st.session_state.metrics = result
                            st.success("✅ Models trained successfully!")
                        else:
                            st.error(f"❌ Error: {result['error']}")

            except Exception as e:
                st.error(f"❌ Exception: {str(e)}")

    # ─── SECTION 2: TRAINING RESULTS ──────────────────────────────────────

    if st.session_state.trained and st.session_state.metrics:
        metrics = st.session_state.metrics

        st.divider()
        st.header("📊 Results")

        # Data statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        stats = metrics["data_statistics"]

        with col1:
            st.metric("Total Samples", stats["total_samples"])
        with col2:
            st.metric("Train Samples", stats["train_samples"])
        with col3:
            st.metric("Test Samples", stats["test_samples"])
        with col4:
            st.metric("Features", stats["features"])
        with col5:
            st.metric("Classes", len(stats["classes"]))

        # Model comparison table
        st.subheader("🏆 Model Comparison")
        model_data = []
        for model_name, model_metrics in metrics["models"].items():
            badge = "🏆" if model_name == metrics["best_model"] else ""
            model_data.append(
                {
                    "Model": f"{badge} {model_name}",
                    "Accuracy": f"{model_metrics['accuracy']:.2%}",
                    "Precision": f"{model_metrics['precision']:.2%}",
                    "Recall": f"{model_metrics['recall']:.2%}",
                    "F1-Score": f"{model_metrics['f1']:.2%}",
                }
            )

        st.dataframe(model_data, use_container_width=True)

        # Confusion matrices
        st.subheader("📈 Confusion Matrices")
        col1, col2, col3 = st.columns(3)

        for idx, (model_name, model_metrics) in enumerate(metrics["models"].items()):
            with [col1, col2, col3][idx]:
                st.write(f"**{model_name}**")
                cm = np.array(model_metrics["confusion_matrix"])
                st.text(f"{cm}")

        # Per-class metrics
        st.subheader("📋 Per-Class Metrics")
        for model_name, model_metrics in metrics["models"].items():
            with st.expander(f"📌 {model_name} - Detailed Metrics"):
                per_class = model_metrics["per_class"]
                class_data = []
                for label, scores in per_class.items():
                    class_data.append(
                        {
                            "Class": label,
                            "Precision": f"{scores['precision']:.4f}",
                            "Recall": f"{scores['recall']:.4f}",
                            "F1-Score": f"{scores['f1']:.4f}",
                            "Support": scores["support"],
                        }
                    )
                st.dataframe(class_data, use_container_width=True)

        # ─── SECTION 3: PREDICTION ────────────────────────────────────────

        st.divider()
        st.header("🔮 Test Predictions")

        col1, col2 = st.columns([2, 1])

        with col1:
            test_text = st.text_area("Enter text to analyze", height=100)

        with col2:
            selected_model = st.selectbox(
                "Select model",
                list(metrics["models"].keys()),
                index=0,
            )

        if st.button("🎯 Predict", use_container_width=True):
            if not test_text.strip():
                st.warning("⚠️ Please enter text to analyze")
            else:
                try:
                    result = st.session_state.pipeline.predict_single(
                        test_text, selected_model
                    )
                    st.success("✅ Prediction successful")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", result["prediction"].upper())
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.2%}",
                        )

                    # Confidence scores
                    st.write("**Confidence Scores:**")
                    scores_df = pd.DataFrame(
                        list(result["scores"].items()), columns=["Class", "Score"]
                    )
                    scores_df["Score"] = scores_df["Score"].apply(
                        lambda x: f"{x:.4f}"
                    )
                    st.dataframe(scores_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # ─── SECTION 4: DOWNLOAD REPORTS ──────────────────────────────────

        st.divider()
        st.header("📥 Download Reports")

        col1, col2 = st.columns(2)

        with col1:
            csv_report = st.session_state.pipeline.generate_report("csv")
            st.download_button(
                label="📄 Download as CSV",
                data=csv_report,
                file_name="sentiment_analysis_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            json_report = st.session_state.pipeline.generate_report("json")
            st.download_button(
                label="📋 Download as JSON",
                data=json_report,
                file_name="sentiment_analysis_report.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
