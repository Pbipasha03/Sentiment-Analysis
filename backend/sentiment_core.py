from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_CANDIDATES = [
    ROOT_DIR / "dataset.csv",
    ROOT_DIR / "sample_dataset_expanded.csv",
    ROOT_DIR.parent / "microtext_sentiment_dataset.csv",
]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def _normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    mapping = {
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
    }
    return mapping.get(label, label)


class TextPreprocessor:
    def __init__(self) -> None:
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.suffixes = ("ingly", "edly", "ing", "edly", "edly", "ed", "ly", "ies", "s")

    def stem(self, token: str) -> str:
        if len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        for suffix in self.suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

    def clean(self, text: object) -> str:
        value = str(text).lower()
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
        value = re.sub(r"@\w+|#\w+", " ", value)
        value = value.translate(PUNCT_TRANSLATION)
        value = re.sub(r"[^a-z\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        tokens = [
            self.stem(token)
            for token in value.split()
            if token and token not in self.stop_words and len(token) > 1
        ]
        return " ".join(tokens)


PREPROCESSOR = TextPreprocessor()


def ensure_texts_payload(payload: Optional[dict]) -> List[str]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be JSON.")
    texts = payload.get("texts")
    if texts is None:
        raise ValueError("Request JSON must include 'texts'.")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        raise ValueError("'texts' must be a list of strings.")
    cleaned = [str(item).strip() for item in texts if str(item).strip()]
    if not cleaned:
        raise ValueError("'texts' must contain at least one non-empty string.")
    return cleaned


def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    columns = {column.lower().strip(): column for column in df.columns}
    text_column = None
    label_column = None

    for name in ("text", "texts", "tweet", "content", "sentence", "review"):
        if name in columns:
            text_column = columns[name]
            break

    for name in ("sentiment", "label", "target", "class"):
        if name in columns:
            label_column = columns[name]
            break

    if not text_column:
        raise ValueError("CSV must include a text column such as 'text'.")

    working = df.copy()
    working[text_column] = working[text_column].astype(str).fillna("").str.strip()
    working = working[working[text_column] != ""].reset_index(drop=True)
    if working.empty:
        raise ValueError("Dataset does not contain any non-empty text rows.")

    if not label_column:
        return pd.DataFrame({"text": working[text_column].astype(str)}), False

    working[label_column] = working[label_column].astype(str).map(_normalize_label)
    working = working[working[label_column] != ""].reset_index(drop=True)
    if working.empty:
        raise ValueError("Dataset does not contain any usable sentiment labels.")

    normalized = pd.DataFrame(
        {
            "text": working[text_column].astype(str),
            "sentiment": working[label_column].astype(str),
        }
    )
    return normalized, True


def load_default_training_dataframe() -> pd.DataFrame:
    for path in DEFAULT_DATASET_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            normalized, has_labels = normalize_dataframe(df)
            if has_labels and len(normalized) >= 3:
                return normalized
    raise FileNotFoundError("No valid default training dataset was found.")


def choose_split(labels: List[str]) -> Tuple[Optional[float], Optional[List[str]]]:
    total_samples = len(labels)
    unique_classes = sorted(set(labels))
    class_count = len(unique_classes)
    min_class_size = min(labels.count(label) for label in unique_classes)

    if total_samples < 6 or class_count < 2 or min_class_size < 2:
        return None, None

    desired_test_count = max(class_count, int(math.ceil(total_samples * 0.2)))
    max_test_count = total_samples - class_count
    test_count = min(desired_test_count, max_test_count)
    if test_count < class_count or test_count <= 0:
        return None, None
    return test_count / total_samples, labels


def format_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@dataclass
class TrainingArtifacts:
    vectorizer: TfidfVectorizer
    models: Dict[str, object]
    metrics: Dict[str, Dict[str, object]]
    best_model_name: str
    labels: List[str]
    data_summary: Dict[str, int]


class SentimentService:
    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.models_trained = False
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.models: Dict[str, object] = {}
        self.metrics: Dict[str, Dict[str, object]] = {}
        self.best_model_name: Optional[str] = None
        self.labels: List[str] = []
        self.data_summary: Dict[str, int] = {}

    def _artifact_path(self, name: str) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir / name

    def save(self) -> None:
        if not self.models_trained or not self.vectorizer or not self.best_model_name:
            return
        joblib.dump(self.vectorizer, self._artifact_path("vectorizer.joblib"))
        joblib.dump(self.models, self._artifact_path("models.joblib"))
        joblib.dump(
            {
                "metrics": self.metrics,
                "best_model_name": self.best_model_name,
                "labels": self.labels,
                "data_summary": self.data_summary,
            },
            self._artifact_path("metadata.joblib"),
        )

    def load(self) -> bool:
        vectorizer_path = self._artifact_path("vectorizer.joblib")
        models_path = self._artifact_path("models.joblib")
        metadata_path = self._artifact_path("metadata.joblib")
        if not (vectorizer_path.exists() and models_path.exists() and metadata_path.exists()):
            return False
        self.vectorizer = joblib.load(vectorizer_path)
        self.models = joblib.load(models_path)
        metadata = joblib.load(metadata_path)
        self.metrics = metadata["metrics"]
        self.best_model_name = metadata["best_model_name"]
        self.labels = metadata["labels"]
        self.data_summary = metadata["data_summary"]
        self.models_trained = bool(self.models and self.vectorizer and self.best_model_name)
        return self.models_trained

    def train(self, df: Optional[pd.DataFrame] = None) -> TrainingArtifacts:
        if df is None:
            df = load_default_training_dataframe()
        normalized_df, has_labels = normalize_dataframe(df)
        if not has_labels:
            raise ValueError("Training data must include both text and sentiment columns.")

        normalized_df["clean_text"] = normalized_df["text"].map(PREPROCESSOR.clean)
        normalized_df = normalized_df[normalized_df["clean_text"] != ""].reset_index(drop=True)
        if len(normalized_df) < 3:
            raise ValueError("Not enough valid text samples after preprocessing.")

        texts = normalized_df["clean_text"].tolist()
        labels = normalized_df["sentiment"].tolist()
        self.labels = sorted(normalized_df["sentiment"].unique().tolist())

        split_size, stratify_labels = choose_split(labels)
        if split_size is None:
            X_train_texts = texts
            X_test_texts = texts
            y_train = labels
            y_test = labels
        else:
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts,
                labels,
                test_size=split_size,
                random_state=42,
                stratify=stratify_labels,
            )

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = self.vectorizer.fit_transform(X_train_texts)
        X_test = self.vectorizer.transform(X_test_texts)

        candidate_models = {
            "MultinomialNB": MultinomialNB(),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "LinearSVC": CalibratedClassifierCV(LinearSVC(random_state=42), cv=3),
        }

        self.models = {}
        self.metrics = {}

        for name, model in candidate_models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = format_metrics(y_test, predictions, self.labels)
            self.models[name] = model
            self.metrics[name] = metrics

        model_preference = {"LinearSVC": 3, "LogisticRegression": 2, "MultinomialNB": 1}
        self.best_model_name = max(
            self.metrics,
            key=lambda model_name: (
                self.metrics[model_name]["f1"],
                self.metrics[model_name]["accuracy"],
                model_preference.get(model_name, 0),
            ),
        )
        self.data_summary = {
            "total_samples": int(len(normalized_df)),
            "train_samples": int(len(X_train_texts)),
            "test_samples": int(len(X_test_texts)),
            "num_classes": int(len(self.labels)),
            "num_features": int(X_train.shape[1]),
        }
        self.models_trained = True
        self.save()

        return TrainingArtifacts(
            vectorizer=self.vectorizer,
            models=self.models,
            metrics=self.metrics,
            best_model_name=self.best_model_name,
            labels=self.labels,
            data_summary=self.data_summary,
        )

    def ensure_trained(self) -> None:
        if self.models_trained:
            return
        if self.load():
            return
        self.train()

    def predict(self, texts: List[str], model_name: Optional[str] = None) -> List[Dict[str, object]]:
        self.ensure_trained()
        if not self.vectorizer or not self.models:
            raise RuntimeError("Models are not trained.")

        selected_model_name = model_name or self.best_model_name
        if selected_model_name not in self.models:
            raise ValueError(f"Unknown model '{selected_model_name}'.")

        model = self.models[selected_model_name]
        cleaned_texts = [PREPROCESSOR.clean(text) for text in texts]
        feature_matrix = self.vectorizer.transform(cleaned_texts)
        predictions = model.predict(feature_matrix)
        probabilities = model.predict_proba(feature_matrix)

        results = []
        for index, original_text in enumerate(texts):
            probability_vector = probabilities[index]
            probability_map = {
                str(label): float(probability_vector[position])
                for position, label in enumerate(model.classes_)
            }
            predicted_label = str(predictions[index])
            results.append(
                {
                    "text": original_text,
                    "clean_text": cleaned_texts[index],
                    "prediction": predicted_label,
                    "confidence": float(probability_map.get(predicted_label, 0.0)),
                    "probabilities": probability_map,
                    "model": selected_model_name,
                }
            )
        return results

    def metrics_payload(self) -> Dict[str, object]:
        self.ensure_trained()
        return {
            "models_trained": self.models_trained,
            "best_model": self.best_model_name,
            "labels": self.labels,
            "metrics": self.metrics,
            "data_summary": self.data_summary,
        }
