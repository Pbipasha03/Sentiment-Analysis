#!/usr/bin/env python3
"""
Production-grade sentiment analysis model training pipeline.
Trains 3 models: Naive Bayes, Logistic Regression, Linear SVM
Handles preprocessing, vectorization, model training, and persistence.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Configuration
BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR
MODEL_DIR = BACKEND_DIR

# Model file paths
VECTORIZER_PATH = MODEL_DIR / 'vectorizer_production.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder_production.pkl'
NB_MODEL_PATH = MODEL_DIR / 'sentiment_nb_production.pkl'
LR_MODEL_PATH = MODEL_DIR / 'sentiment_lr_production.pkl'
SVM_MODEL_PATH = MODEL_DIR / 'sentiment_svm_production.pkl'
METRICS_PATH = MODEL_DIR / 'model_metrics_production.json'

# Global instances
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preserve negation words
NEGATION_WORDS = {'not', 'no', 'nor', 'neither', 'never', 'nothing', 'nowhere', 'nobody'}
stop_words = stop_words - NEGATION_WORDS


def generate_large_dataset():
    """Generate a large, balanced dataset if original is too small."""
    positive_texts = [
        "I love this product, it's amazing!", "Excellent quality and service!",
        "Best purchase ever, highly recommended!", "Outstanding experience!",
        "This is fantastic, I'm very happy!", "Wonderful product, exceeded expectations!",
        "Absolutely love it, best thing ever!", "Perfect! Five stars!",
        "Incredible service and great quality!", "Amazing! Very satisfied!",
        "This made my day, it's so good!", "Best value for money!",
        "Love love love this product!", "Can't ask for better!",
        "Simply the best, period!", "Stunning quality and design!",
        "Highly satisfied, would buy again!", "Absolutely brilliant!",
        "This is worth every penny!", "Fantastic experience overall!",
        "Great stuff, very happy!", "Love the quality!", "Best decision ever!",
        "Excellent in every way!", "Really impressed!", "Super happy!",
        "This is brilliant!", "Couldn't be happier!", "Love it so much!",
        "Amazing product!", "Best thing ever!", "Fantastic quality!",
        "Great purchase!", "Very pleased!", "Love this!",
        "Awesome!", "Perfect!", "Great!", "Excellent!", "Wonderful!",
        "Fantastic improvement!", "Best service ever!", "Highly impressed!",
        "Outstanding quality!", "Brilliant experience!", "Best choice!",
        "Beautifully made!", "Excellent design!", "Quality is top notch!",
    ]

    negative_texts = [
        "Terrible quality, very disappointed!", "Awful experience, waste of money!",
        "Horrible product, don't buy!", "This is the worst ever!",
        "Complete disaster, very unhappy!", "Awful service and poor quality!",
        "Absolutely terrible, total waste!", "Disgusting, not recommend!",
        "One of the worst purchases!", "Extremely poor quality!",
        "This is a scam, very bad!", "Worst experience of my life!",
        "Horrible, never again!", "Terrible quality and service!",
        "This failed immediately!", "Complete waste of time and money!",
        "Bad product, bad service!", "Don't waste your money on this!",
        "Awful, couldn't be worse!", "Garbage product!",
        "Not satisfied at all!", "Very poor quality!",
        "This is broken!", "Defective product!", "Bad quality!",
        "Terrible!", "Horrible!", "Awful!", "Awful experience!",
        "Poor quality!", "Not good!", "Bad!", "Hate it!",
        "Disappointing!", "Not worth it!", "Disaster!",
        "Terrible service!", "Horrible experience!", "Bad choice!",
        "Cheap quality!", "Defective item!", "Broken product!",
        "Unhappy!", "Frustrated!", "Annoyed!", "Disgusted!",
    ]

    neutral_texts = [
        "It's okay, nothing special.", "Average product.", "Neither good nor bad.",
        "It's fine I guess.", "Mediocre quality.", "Just okay, nothing more.",
        "It works, that's it.", "Standard, nothing unusual.", "Acceptable quality.",
        "Not great, not terrible.", "It's alright.", "Could be better, could be worse.",
        "Decent enough.", "Average experience.", "Ordinary product.",
        "So-so quality.", "It's fine.", "Normal." , "Basic product.",
        "Adequate.", "Satisfactory." , "Fair quality.", "It does the job.",
        "Reasonable.", "Nothing remarkable.", "Common product.", "Standard quality.",
        "It works as described.", "Acceptable." , "Tolerable.", "Manageable.",
        "Somewhat good.", "Somewhat bad.", "In between.", "Neutral experience.",
        "Nothing special here.", "Typical product.", "Expected quality.",
        "Fair enough.", "Decent product.", "Normal experience!", "Basic quality!",
    ]

    # Create balanced dataset
    data = []
    
    # Add 400 samples of each class for ~1200 total
    for text in positive_texts * 13:
        data.append({'text': text, 'sentiment': 'Positive'})
    
    for text in negative_texts * 13:
        data.append({'text': text, 'sentiment': 'Negative'})
    
    for text in neutral_texts * 13:
        data.append({'text': text, 'sentiment': 'Neutral'})

    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return df


def load_or_create_dataset():
    """Load dataset from file or create large one."""
    dataset_paths = [
        DATA_DIR / 'microtext_sentiment_dataset.csv',
        DATA_DIR / 'dataset.csv',
        DATA_DIR / 'sample_dataset_expanded.csv',
    ]

    df = None
    for path in dataset_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                print(f"✓ Loaded dataset from {path.name} ({len(df)} samples)")
                break
            except Exception as e:
                print(f"✗ Failed to load {path.name}: {e}")

    # If dataset is too small or missing, generate a large one
    if df is None or len(df) < 100:
        print("⚠ Dataset too small or missing. Generating large balanced dataset...")
        df = generate_large_dataset()
        print(f"✓ Generated dataset with {len(df)} samples")

    # Validate columns
    if 'text' not in df.columns:
        if df.shape[1] >= 1:
            df.rename(columns={df.columns[0]: 'text'}, inplace=True)
        else:
            raise ValueError("Dataset must have a 'text' column")

    if 'sentiment' not in df.columns:
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]
        if sentiment_cols:
            df.rename(columns={sentiment_cols[0]: 'sentiment'}, inplace=True)
        else:
            if df.shape[1] >= 2:
                df.rename(columns={df.columns[1]: 'sentiment'}, inplace=True)

    # Clean data
    df = df.dropna(subset=['text', 'sentiment'])
    df['text'] = df['text'].astype(str).str.strip()
    df['sentiment'] = df['sentiment'].astype(str).str.strip()

    # Normalize sentiment labels
    df['sentiment'] = df['sentiment'].str.lower()
    df['sentiment'] = df['sentiment'].replace({
        'pos': 'positive', 'positive': 'positive',
        'neg': 'negative', 'negative': 'negative',
        'neu': 'neutral', 'neutral': 'neutral'
    })

    # Filter to valid sentiments
    valid_sentiments = {'positive', 'negative', 'neutral'}
    df = df[df['sentiment'].isin(valid_sentiments)]

    print(f"✓ Loaded and cleaned {len(df)} samples")
    print(f"  Class distribution:")
    print(df['sentiment'].value_counts().to_string().replace('\n', '\n  '))

    return df


def clean_text(text):
    """Clean text: lowercase, remove URLs/mentions/hashtags, remove punctuation, lemmatize."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word) > 2])
    return text.strip()


def train_models(df, test_size=0.2, random_state=42):
    """Train 3 sentiment models and save them."""
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)

    # Step 1: Encode labels
    print("\n[1] Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])
    print(f"✓ Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Step 2: Clean text
    print("\n[2] Cleaning text data...")
    X_clean = df['text'].apply(clean_text)
    print(f"✓ Text cleaned, example: '{df['text'].iloc[0][:50]}...' -> '{X_clean.iloc[0][:50]}...'")

    # Step 3: Train-test split (stratified)
    print("\n[3] Splitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")

    # Step 4: TF-IDF Vectorization (fit ONLY on training data)
    print("\n[4] TF-IDF Vectorization (fit on training only)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words=None,  # Already removed in clean_text
        lowercase=False  # Already lowercased in clean_text
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✓ Vectorizer fit on training data")
    print(f"✓ Features: {X_train_vec.shape[1]}")

    # Step 5: Train models
    print("\n[5] Training 3 models...")
    models = {}

    print("  - Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_vec, y_train)
    models['Naive Bayes'] = nb_model
    print("    ✓ Trained")

    print("  - Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    lr_model.fit(X_train_vec, y_train)
    models['Logistic Regression'] = lr_model
    print("    ✓ Trained")

    print("  - Training Linear SVM...")
    svm_model = LinearSVC(max_iter=2000, class_weight='balanced', random_state=random_state, dual=False)
    svm_model.fit(X_train_vec, y_train)
    models['SVM'] = svm_model
    print("    ✓ Trained")

    # Step 6: Evaluate models
    print("\n[6] Evaluating models...")
    metrics = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        }

        print(f"\n  {name}:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")

    # Step 7: Save models
    print("\n[7] Saving models...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"✓ Vectorizer saved to {VECTORIZER_PATH.name}")

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"✓ Label encoder saved to {LABEL_ENCODER_PATH.name}")

    joblib.dump(models['Naive Bayes'], NB_MODEL_PATH)
    print(f"✓ Naive Bayes model saved to {NB_MODEL_PATH.name}")

    joblib.dump(models['Logistic Regression'], LR_MODEL_PATH)
    print(f"✓ Logistic Regression model saved to {LR_MODEL_PATH.name}")

    joblib.dump(models['SVM'], SVM_MODEL_PATH)
    print(f"✓ SVM model saved to {SVM_MODEL_PATH.name}")

    # Step 8: Save metrics as JSON
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {METRICS_PATH.name}")

    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE - All models trained successfully!")
    print("="*60 + "\n")

    return {
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'models': models,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'X_test_vec': X_test_vec,
        'y_test_pred': {name: model.predict(X_test_vec) for name, model in models.items()}
    }


if __name__ == '__main__':
    # Load or create dataset
    df = load_or_create_dataset()

    # Train models
    results = train_models(df)

    print("\n✓ Ready to use in Streamlit app or API!")
    sys.exit(0)
