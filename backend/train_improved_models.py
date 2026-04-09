"""
train_improved_models.py
========================
Improved sentiment analysis model training with:
- Proper negation handling (keeping "not", "neither", "nor")
- N-grams for phrase capture (e.g., "not good")
- Class balancing for better neutral classification
- Neutral handling with confidence thresholds
- Comprehensive evaluation metrics

HOW TO USE:
    python train_improved_models.py

OUTPUT:
    - vectorizer_improved.pkl (TF-IDF vectorizer with negation support)
    - sentiment_nb_improved.pkl (Naive Bayes model)
    - sentiment_lr_improved.pkl (Logistic Regression model)
    - sentiment_svm_improved.pkl (SVM model)
    - sentiment_metrics_improved.json (training metrics)
"""

import json
import pickle
import re
from pathlib import Path

import nltk
import pandas as pd
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
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Download resources
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

ROOT_DIR = Path(__file__).resolve().parent
# Try multiple dataset locations
DATASET_PATHS = [
    ROOT_DIR / "dataset.csv",  # Preferred: more samples
    ROOT_DIR / "sample_dataset_expanded.csv",
    ROOT_DIR.parent / "microtext_sentiment_dataset.csv"
]
DATASET_PATH = next((p for p in DATASET_PATHS if p.exists()), None)
VECTORIZER_PATH = ROOT_DIR / "vectorizer_improved.pkl"
NB_MODEL_PATH = ROOT_DIR / "sentiment_nb_improved.pkl"
LR_MODEL_PATH = ROOT_DIR / "sentiment_lr_improved.pkl"
SVM_MODEL_PATH = ROOT_DIR / "sentiment_svm_improved.pkl"
METRICS_PATH = ROOT_DIR / "sentiment_metrics_improved.json"

LABELS = ["negative", "neutral", "positive"]

# ─── IMPROVED PREPROCESSING ───────────────────────────────────────────────────
# Keep important negation words for better sentiment detection

def get_improved_stopwords():
    """
    Get stopwords but KEEP negation words which are important for sentiment.
    Examples:
      - "not good" should be captured (negated positive)
      - "neither good nor bad" should be captured (explicit neutral)
    """
    stop_words = set(stopwords.words("english"))
    
    # Remove negation and related words from stopwords
    keep_words = {
        "not", "no", "nor", "neither", "never", "nothing", "nowhere",
        "nobody", "isn", "aren", "wasn", "weren", "hasn", "hadn",
        "haven", "doesnt", "didnt", "dont", "should", "shouldnt",
        "wont", "wouldnt", "cant", "couldnt", "shan", "shant"
    }
    
    # Also keep common negation patterns
    stop_words = stop_words - keep_words
    
    return stop_words


STOP_WORDS_IMPROVED = get_improved_stopwords()


def preprocess_improved(text):
    """
    Enhanced preprocessing that preserves negation and important phrases.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove special characters (but keep spaces)
    4. Remove extra whitespace
    5. Tokenize and remove stopwords (excluding negations)
    6. Keep words with length > 2
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize and filter stopwords
    # Important: we keep negation words like "not", "neither", "nor"
    tokens = [w for w in text.split() if w not in STOP_WORDS_IMPROVED and len(w) > 2]
    
    return " ".join(tokens)


def balance_dataset(texts, labels):
    """
    Balance dataset to have equal samples per class.
    This helps prevent bias toward any particular sentiment.
    """
    df = pd.DataFrame({"text": texts, "label": labels})
    
    # Find minimum class size
    class_counts = df["label"].value_counts()
    print(f"\n📊 Original class distribution:")
    for label in LABELS:
        count = class_counts.get(label, 0)
        print(f"   {label:12s}: {count:3d} samples")
    
    min_samples = class_counts.min()
    
    # Balance dataset to have min_samples per class
    balanced_data = []
    for label in LABELS:
        label_data = df[df["label"] == label].sample(n=min_samples, random_state=42)
        balanced_data.append(label_data)
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Balanced class distribution:")
    balanced_counts = balanced_df["label"].value_counts()
    for label in LABELS:
        count = balanced_counts.get(label, 0)
        print(f"   {label:12s}: {count:3d} samples")
    
    return balanced_df["text"].tolist(), balanced_df["label"].tolist()


def normalize_label(label):
    """Normalize label names to lowercase."""
    if isinstance(label, str):
        label = label.strip().lower()
        if label in ["pos", "positive", "1"]:
            return "positive"
        elif label in ["neg", "negative", "0"]:
            return "negative"
        elif label in ["neu", "neutral", "2"]:
            return "neutral"
    return "neutral"


def rule_based_neutral_detection(text):
    """
    Rule-based detection for explicit neutral phrases.
    Returns (is_neutral, confidence) if detected, else (False, 0.0)
    """
    text_lower = text.lower()
    
    neutral_phrases = [
        "neither.*nor", "neither good nor bad", "average", "okay",
        "so so", "not bad not good", "just okay", "moderate",
        "alright", "fine", "decent", "nothing special", "not special"
    ]
    
    for phrase in neutral_phrases:
        if re.search(phrase, text_lower):
            return True, 0.95  # High confidence for explicit neutral
    
    return False, 0.0


def main():
    print("\n" + "="*70)
    print("🚀 IMPROVED SENTIMENT MODEL TRAINING")
    print("="*70)
    
    # ─── LOAD DATASET ──────────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("   Expected columns: text, sentiment")
        return
    
    df = pd.read_csv(DATASET_PATH)
    
    # Handle both "sentiment" and "label" column names
    if "sentiment" in df.columns:
        label_col = "sentiment"
    elif "label" in df.columns:
        label_col = "label"
    else:
        print(f"❌ Dataset must have either 'sentiment' or 'label' column")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    df = df.dropna(subset=["text", label_col]).reset_index(drop=True)
    
    # Normalize labels
    df["label"] = df[label_col].apply(normalize_label)
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    
    # ─── PREPROCESS TEXTS ──────────────────────────────────────────────────
    print("\n⚙️  Preprocessing texts...")
    df["text_clean"] = df["text"].apply(preprocess_improved)
    df = df[df["text_clean"].str.len() > 0]  # Remove empty texts after cleaning
    
    texts = df["text"].tolist()
    texts_clean = df["text_clean"].tolist()
    labels = df["label"].tolist()
    
    print(f"✅ Preprocessed {len(texts)} texts")
    print(f"   Sample cleaned text: '{texts_clean[0]}'")
    
    # ─── BALANCE DATASET ───────────────────────────────────────────────────
    print("\n⚖️  Balancing dataset for equal class distribution...")
    texts_clean, labels = balance_dataset(texts_clean, labels)
    
    # ─── SPLIT DATA ────────────────────────────────────────────────────────
    print("\n📊 Splitting data (80/20)...")
    X_train_clean, X_test_clean, y_train, y_test = train_test_split(
        texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"✅ Train set: {len(X_train_clean)} samples")
    print(f"✅ Test set:  {len(X_test_clean)} samples")
    
    # ─── FEATURE ENGINEERING ──────────────────────────────────────────────
    print("\n🔤 Vectorizing text (TF-IDF with n-grams)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Capture unigrams and bigrams
        min_df=1,              # Minimum document frequency
        max_df=0.95,           # Maximum document frequency (95%)
        sublinear_tf=True      # Sublinear term frequency scaling
    )
    
    X_train = vectorizer.fit_transform(X_train_clean)
    X_test = vectorizer.transform(X_test_clean)
    
    print(f"✅ Vectorizer created")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Training shapes: {X_train.shape}")
    print(f"   Test shapes: {X_test.shape}")
    print(f"   Sample n-grams: {vectorizer.get_feature_names_out()[:10]}")
    
    # ─── COMPUTE CLASS WEIGHTS ────────────────────────────────────────────
    print("\n⚖️  Computing class weights for balanced training...")
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(LABELS),
        y=np.array(y_train)
    )
    class_weight_dict = dict(zip(LABELS, class_weights))
    
    print(f"✅ Class weights:")
    for label, weight in class_weight_dict.items():
        print(f"   {label:12s}: {weight:.3f}")
    
    # ─── TRAIN MODELS ─────────────────────────────────────────────────────
    print("\n🤖 Training models with class balancing...")
    
    # 1. Multinomial Naive Bayes (baseline)
    print("\n   1️⃣  Naive Bayes...")
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"      ✅ Accuracy: {acc_nb:.4f}")
    
    # 2. Logistic Regression with class weighting
    print("\n   2️⃣  Logistic Regression (with class weights)...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight=class_weight_dict,
        C=1.0,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"      ✅ Accuracy: {acc_lr:.4f}")
    
    # 3. Support Vector Machine with class weighting
    print("\n   3️⃣  Support Vector Machine (with class weights)...")
    svm = LinearSVC(
        max_iter=2000,
        random_state=42,
        class_weight=class_weight_dict,
        C=1.0
    )
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"      ✅ Accuracy: {acc_svm:.4f}")
    
    # ─── EVALUATE MODELS ──────────────────────────────────────────────────
    print("\n📈 Evaluating models...")
    
    def evaluate_model(name, model, y_pred, y_proba=None):
        print(f"\n   {name}:")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        print(f"      Accuracy:  {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall:    {rec:.4f}")
        print(f"      F1 Score:  {f1:.4f}")
        
        # Per-class metrics (IMPORTANT for Neutral)
        print(f"\n      Per-class metrics:")
        report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0, output_dict=True)
        for label in LABELS:
            metrics = report[label]
            print(f"         {label:12s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        print(f"\n      Confusion Matrix ({' | '.join(LABELS)}):")
        for i, label in enumerate(LABELS):
            print(f"         {label:12s}: {cm[i]}")
        
        return {
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": {label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"])
            } for label in LABELS}
        }
    
    metrics_nb = evaluate_model("Naive Bayes", nb, y_pred_nb)
    metrics_lr = evaluate_model("Logistic Regression", lr, y_pred_lr)
    metrics_svm = evaluate_model("SVM", svm, y_pred_svm)
    
    # ─── SELECT BEST MODEL ────────────────────────────────────────────────
    print("\n🏆 Model Selection:")
    models_results = [
        ("Naive Bayes", acc_nb, metrics_nb),
        ("Logistic Regression", acc_lr, metrics_lr),
        ("SVM", acc_svm, metrics_svm)
    ]
    best_name, best_acc, best_metrics = max(models_results, key=lambda x: x[1])
    print(f"   Best model: {best_name} ({best_acc:.4f} accuracy)")
    
    # ─── SAVE MODELS ──────────────────────────────────────────────────────
    print("\n💾 Saving models...")
    
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"   ✅ {VECTORIZER_PATH.name}")
    
    with open(NB_MODEL_PATH, "wb") as f:
        pickle.dump(nb, f)
    print(f"   ✅ {NB_MODEL_PATH.name}")
    
    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    print(f"   ✅ {LR_MODEL_PATH.name}")
    
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(svm, f)
    print(f"   ✅ {SVM_MODEL_PATH.name}")
    
    # ─── SAVE METRICS ────────────────────────────────────────────────────
    metrics_summary = {
        "dataset_size": len(texts),
        "train_size": len(X_train_clean),
        "test_size": len(X_test_clean),
        "labels": LABELS,
        "class_weights": class_weight_dict,
        "preprocessing": "improved (negation preserved, n-grams)",
        "vectorizer_config": {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95
        },
        "models": [metrics_nb, metrics_lr, metrics_svm],
        "best_model": best_name,
        "best_accuracy": float(best_acc)
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"   ✅ {METRICS_PATH.name}")
    
    # ─── DEMONSTRATION ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("🧪 TESTING ON SAMPLE TEXTS")
    print("="*70)
    
    test_texts = [
        "I love this product, it's amazing!",
        "This is terrible, worst experience ever",
        "It is neither good nor bad",
        "Not bad, but not great either",
        "Really happy with results!",
        "Average quality, okay for the price",
        "I hate it!",
        "It's fine, satisfied with it"
    ]
    
    for text in test_texts:
        clean = preprocess_improved(text)
        
        # Rule-based neutral detection
        is_neutral_rule, conf_rule = rule_based_neutral_detection(text)
        
        # Vectorize
        X = vectorizer.transform([clean])
        
        # Get predictions with probabilities
        if hasattr(lr, "predict_proba"):
            proba = lr.predict_proba(X)[0]
            pred_lr = lr.predict(X)[0]
            confidence_lr = proba[list(lr.classes_).index(pred_lr)]
        else:
            pred_lr = lr.predict(X)[0]
            confidence_lr = 0.0
        
        pred_nb = nb.predict(X)[0]
        
        # Apply neutral threshold logic
        if confidence_lr < 0.4:  # Low confidence → neutral
            final_label = "neutral"
            reason = "(low confidence threshold)"
        elif is_neutral_rule:
            final_label = "neutral"
            reason = "(rule-based detection)"
        else:
            final_label = pred_lr
            reason = ""
        
        print(f"\n{'─'*70}")
        print(f"📝 Text: {text}")
        print(f"   Cleaned: {clean}")
        print(f"   NB Pred: {pred_nb}")
        print(f"   LR Pred: {pred_lr} (confidence: {confidence_lr:.3f})")
        print(f"   Final:   {final_label} {reason}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📢 NOTE: Update app.py to use improved models:")
    print(f"   - Import from: vectorizer_improved.pkl, sentiment_*_improved.pkl")
    print(f"   - Use improved preprocessing function")
    print(f"   - Apply confidence thresholds + rule-based neutral detection")
    print("\n")


if __name__ == "__main__":
    main()
