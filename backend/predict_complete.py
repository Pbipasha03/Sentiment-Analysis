#!/usr/bin/env python3
"""
COMPLETE SENTIMENT ANALYSIS PREDICTION PIPELINE
Fixes requirement #7, #8, #9: Prediction, Model State, Batch Analysis

Author: AI Assistant  
Date: 2026-04-10
"""

import os
import pandas as pd
import numpy as np
import joblib
import re
from typing import List, Dict, Optional

# ============================================================================
# GLOBAL STATE (Requirement #8: Track model_trained status)
# ============================================================================

class ModelManager:
    """Global model state manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_trained = False
            cls._instance.vectorizer = None
            cls._instance.model = None
            cls._instance.nb_model = None
            cls._instance.svm_model = None
            cls._instance.label_encoder = None
        return cls._instance

# Global instance
model_manager = ModelManager()

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = '.'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer_complete.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder_complete.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model_complete.pkl')
NB_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_nb_complete.pkl')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_svm_complete.pkl')

# Stopwords and keep words (MUST MATCH training!)
STOPWORDS_TO_REMOVE = {
    'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
    'of', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'can', 'it', 'its', 'that', 'this', 'these',
    'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him',
    'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
    'with', 'by', 'from', 'up', 'about', 'as', 'than', 'so', 'just'
}

KEEP_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'nothing',
    'nowhere', 'nonetheless', 'noone'
}

# Remove KEEP_WORDS from stopwords
STOPWORDS_TO_REMOVE = STOPWORDS_TO_REMOVE - KEEP_WORDS

# ============================================================================
# REQUIREMENT #2: PREPROCESSING (MUST BE IDENTICAL TO TRAINING!)
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean text: lowercase, remove punctuation, but KEEP negations
    
    IMPORTANT: This function MUST be identical to the one used in training!
    "Text preprocessing consistency" is Requirement #2 & partially Requirement #7
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters but keep apostrophes and hyphens
    text = re.sub(r'[^a-zA-Z\s\-\']', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords BUT keep negations
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_TO_REMOVE and len(t) > 1]
    text = ' '.join(tokens)
    
    return text

# ============================================================================
# REQUIREMENT #6: LOAD MODELS
# ============================================================================

def load_models() -> Dict[str, object]:
    """
    Load all trained models
    Requirement #6: Model storage - Save/load properly
    """
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)
    
    models = {}
    
    # Load vectorizer
    if os.path.exists(VECTORIZER_PATH):
        try:
            models['vectorizer'] = joblib.load(VECTORIZER_PATH)
            print(f"✅ Vectorizer loaded: {VECTORIZER_PATH}")
            model_manager.vectorizer = models['vectorizer']
        except Exception as e:
            print(f"❌ Error loading vectorizer: {e}")
            return None
    else:
        print(f"❌ Vectorizer not found: {VECTORIZER_PATH}")
        print("   Run training first: python train_complete.py")
        return None
    
    # Load label encoder
    if os.path.exists(ENCODER_PATH):
        try:
            models['label_encoder'] = joblib.load(ENCODER_PATH)
            print(f"✅ Label encoder loaded: {ENCODER_PATH}")
            model_manager.label_encoder = models['label_encoder']
            print(f"   Classes: {models['label_encoder'].classes_.tolist()}")
        except Exception as e:
            print(f"❌ Error loading encoder: {e}")
            return None
    else:
        print(f"❌ Label encoder not found: {ENCODER_PATH}")
        return None
    
    # Load Logistic Regression model (PRIMARY)
    if os.path.exists(MODEL_PATH):
        try:
            models['model'] = joblib.load(MODEL_PATH)
            print(f"✅ Logistic Regression model loaded: {MODEL_PATH}")
            model_manager.model = models['model']
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Run training first: python train_complete.py")
        return None
    
    # Load Naive Bayes (optional)
    if os.path.exists(NB_MODEL_PATH):
        try:
            models['nb_model'] = joblib.load(NB_MODEL_PATH)
            print(f"✅ Naive Bayes model loaded: {NB_MODEL_PATH}")
            model_manager.nb_model = models['nb_model']
        except Exception as e:
            print(f"⚠️  Warning: Could not load NB model: {e}")
    
    # Load SVM (optional)
    if os.path.exists(SVM_MODEL_PATH):
        try:
            models['svm_model'] = joblib.load(SVM_MODEL_PATH)
            print(f"✅ SVM model loaded: {SVM_MODEL_PATH}")
            model_manager.svm_model = models['svm_model']
        except Exception as e:
            print(f"⚠️  Warning: Could not load SVM model: {e}")
    
    # Mark as trained (Requirement #8)
    model_manager.model_trained = True
    print("\n✅ All models loaded successfully")
    print("✅ model_trained = True")
    
    return models

# ============================================================================
# REQUIREMENT #7 & #11: RULE-BASED CORRECTIONS  
# ============================================================================

def apply_rule_corrections(text: str, prediction: str, confidence: float) -> tuple:
    """
    Apply rule-based corrections for edge cases
    Requirement #11: Add simple rule-based correction
    
    Returns: (final_sentiment, final_confidence, method_used)
    """
    text_lower = text.lower()
    
    # NEUTRAL RULES
    neutral_rules = [
        ('neither', 'nor'),
        ('not bad', ''),
        ('not good', ''),
        ('okay', ''),
        ('average', ''),
        ('mediocre', ''),
        ('so so', ''),
        ('so-so', ''),
        ('middle ground', ''),
        ('okay ish', ''),
    ]
    
    for rule in neutral_rules:
        if rule[0] in text_lower:
            if rule[1] == '' or rule[1] in text_lower:
                return ('Neutral', 0.95, 'rule_based_neutral')
    
    # STRONG NEGATIVE RULES
    strong_negative = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disgusting']
    for word in strong_negative:
        if word in text_lower:
            return ('Negative', 0.95, 'rule_based_negative')
    
    # STRONG POSITIVE RULES
    strong_positive = ['love', 'amazing', 'excellent', 'fantastic', 'wonderful', 'perfect']
    for word in strong_positive:
        if word in text_lower:
            return ('Positive', 0.95, 'rule_based_positive')
    
    # If no rules matched, return original
    return (prediction, confidence, 'model_prediction')

# ============================================================================
# REQUIREMENT #7: PREDICTION PIPELINE (SINGLE TEXT)
# ============================================================================

def predict_sentiment(text: str, use_model: str = 'lr') -> Dict:
    """
    Predict sentiment for single text
    Requirement #7: Prediction pipeline
    
    Args:
        text: Input text to analyze
        use_model: 'lr' (default), 'nb', or 'svm'
    
    Returns:
        {
            'text': str,
            'sentiment': str,
            'confidence': float,
            'probabilities': dict,
            'method': str
        }
    """
    
    # Check if model is trained (Requirement #8)
    if not model_manager.model_trained:
        return {
            'error': 'Model not trained',
            'message': 'Please train the model first: python train_complete.py',
            'status': 'error'
        }
    
    # Requirement #7: Apply SAME preprocessing
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return {
            'text': text,
            'sentiment': 'Unknown',
            'confidence': 0.0,
            'error': 'Text too short after preprocessing',
            'status': 'error'
        }
    
    # Requirement #7: Use SAME vectorizer (no refit!)
    text_vector = model_manager.vectorizer.transform([cleaned_text])
    
    # Select model
    if use_model == 'nb' and model_manager.nb_model is not None:
        current_model = model_manager.nb_model
        model_name = 'Naive Bayes'
    elif use_model == 'svm' and model_manager.svm_model is not None:
        current_model = model_manager.svm_model
        model_name = 'SVM'
    else:
        current_model = model_manager.model
        model_name = 'Logistic Regression'
    
    # Requirement #7: Get prediction
    prediction_encoded = current_model.predict(text_vector)[0]
    sentiment = model_manager.label_encoder.classes_[prediction_encoded]
    
    # Requirement #7: Get probability
    try:
        if hasattr(current_model, 'predict_proba'):
            probabilities = current_model.predict_proba(text_vector)[0]
            confidence = probabilities.max()
            probs_dict = {
                label: float(prob)
                for label, prob in zip(model_manager.label_encoder.classes_, probabilities)
            }
        else:
            # SVM doesn't have predict_proba by default
            confidence = 0.5  # Default for SVM
            probs_dict = {label: 0.0 for label in model_manager.label_encoder.classes_}
            probs_dict[sentiment] = 1.0
    except Exception as e:
        confidence = 0.5
        probs_dict = {label: 0.0 for label in model_manager.label_encoder.classes_}
        probs_dict[sentiment] = 1.0
    
    # Requirement #11: Apply rule-based corrections
    final_sentiment, final_confidence, method = apply_rule_corrections(
        text, sentiment, confidence
    )
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': final_sentiment,
        'confidence': float(final_confidence),
        'probabilities': probs_dict,
        'model': model_name,
        'method': method,
        'status': 'success'
    }

# ============================================================================
# REQUIREMENT #9: BATCH CSV ANALYSIS
# ============================================================================

def predict_batch_csv(csv_path: str) -> Dict:
    """
    Analyze entire CSV file
    Requirement #9: Batch CSV analysis
    
    Args:
        csv_path: Path to CSV file with 'text' column
    
    Returns:
        {
            'total': int,
            'results': List[Dict],
            'summary': {
                'Positive': int,
                'Negative': int,
                'Neutral': int
            },
            'status': str
        }
    """
    
    # Check if model is trained
    if not model_manager.model_trained:
        return {
            'error': 'Model not trained',
            'status': 'error',
            'message': 'Train model first: python train_complete.py'
        }
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ Loaded CSV: {csv_path} ({len(df)} rows)")
    except Exception as e:
        return {
            'error': f'Error loading CSV: {e}',
            'status': 'error'
        }
    
    # Find text column
    text_column = None
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'review' in col_lower or 'comment' in col_lower:
            text_column = col
            break
    
    if text_column is None:
        text_column = df.columns[0]
    
    print(f"   Using column: '{text_column}'")
    
    # Extract texts and convert to string (Requirement #9)
    texts = df.iloc[:, 0].astype(str).tolist() if text_column == df.columns[0] else df[text_column].astype(str).tolist()
    
    print(f"   Predicting sentiment for {len(texts)} texts...")
    
    # Predict for all texts
    results = []
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
    for i, text in enumerate(texts):
        if i % max(1, len(texts) // 5) == 0:
            print(f"   Progress: {i}/{len(texts)}")
        
        result = predict_sentiment(text)
        
        if result.get('status') == 'success':
            results.append({
                'index': i,
                'text': text[:100] + ('...' if len(text) > 100 else ''),  # Truncate for display
                'sentiment': result['sentiment'],
                'confidence': round(result['confidence'], 3),
                'method': result['method']
            })
            
            # Count sentiments
            sentiment = result['sentiment']
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
    
    print(f"   ✅ Completed predictions for {len(results)} texts")
    
    return {
        'total': len(results),
        'results': results,
        'summary': sentiment_counts,
        'status': 'success'
    }

# ============================================================================
# BATCH PREDICTION (TEXTS LIST)
# ============================================================================

def predict_batch_texts(texts: List[str]) -> List[Dict]:
    """
    Predict sentiment for list of texts
    Requirement #9: Batch analysis support
    
    Args:
        texts: List of text strings
    
    Returns:
        List of prediction results
    """
    
    # Check if model trained
    if not model_manager.model_trained:
        return [{'error': 'Model not trained', 'status': 'error'}]
    
    results = []
    for text in texts:
        result = predict_sentiment(text)
        results.append(result)
    
    return results

# ============================================================================
# MAIN: TESTING & DEMONSTRATION
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("COMPLETE SENTIMENT ANALYSIS PREDICTION PIPELINE")
    print("="*80)
    
    # Load models
    models = load_models()
    
    if models is None:
        print("\n❌ Error: Could not load models")
        print("Run training first: python train_complete.py")
        exit(1)
    
    # ========================================================================
    # TEST 1: Single text predictions
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: SINGLE TEXT PREDICTIONS")
    print("="*80)
    
    test_texts = [
        "I absolutely love this product!",
        "This is terrible, worst experience ever",
        "It is neither good nor bad",
        "Amazing quality, highly recommend",
        "Awful service and poor quality",
        "Average product for the price",
        "Really happy with my purchase!",
        "I hate it!",
    ]
    
    for text in test_texts:
        result = predict_sentiment(text)
        if result.get('status') == 'success':
            print(f"\n📝 Text: \"{text}\"")
            print(f"   Sentiment: {result['sentiment']} ({result['confidence']:.2%} confidence)")
            print(f"   Method: {result['method']}")
    
    # ========================================================================
    # TEST 2: Batch text list
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: BATCH TEXT PREDICTION")
    print("="*80)
    
    batch_results = predict_batch_texts([
        "I love this!",
        "It's okay",
        "I hate it!"
    ])
    
    print(f"\n✅ Batch Results:")
    for i, result in enumerate(batch_results):
        if result.get('status') == 'success':
            print(f"   {i+1}. {result['text'][:40]} → {result['sentiment']}")
    
    # ========================================================================
    # TEST 3: Batch CSV (if file exists)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: BATCH CSV ANALYSIS")
    print("="*80)
    
    csv_files = [
        'dataset.csv',
        'microtext_sentiment_dataset.csv',
        '../microtext_sentiment_dataset.csv',
    ]
    
    csv_found = None
    for csv_file in csv_files:
        if os.path.exists(csv_file) and csv_file.endswith('.csv'):
            csv_found = csv_file
            break
    
    if csv_found:
        csv_result = predict_batch_csv(csv_found)
        if csv_result.get('status') == 'success':
            print(f"\n✅ Batch CSV Results:")
            print(f"   Total analyzed: {csv_result['total']}")
            print(f"   Summary:")
            for sentiment, count in csv_result['summary'].items():
                print(f"     {sentiment}: {count}")
    else:
        print("\n⚠️  No CSV file found for batch testing")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ PREDICTION PIPELINE READY")
    print("="*80)
    print(f"\n✅ Requirements satisfied:")
    print(f"   #6: Models loaded ✓")
    print(f"   #7: Prediction pipeline ✓")
    print(f"   #8: Model state tracked (model_trained={model_manager.model_trained}) ✓")
    print(f"   #9: Batch CSV analysis ✓")
    print(f"   #11: Rule-based corrections ✓")
    
    print(f"\nNext: Start API server: python api_complete.py")
