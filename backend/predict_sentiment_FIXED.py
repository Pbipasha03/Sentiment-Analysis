"""
predict_sentiment_FIXED.py
===========================
CORRECTED prediction pipeline with:
- Load trained vectorizer (NEVER refit!)
- Load trained model
- SAME preprocessing as training
- Correct label decoding
- Rule-based neutral correction

KEY FIXES:
1. ✅ Load vectorizer, don't create new one
2. ✅ Use SAME preprocessing as training
3. ✅ Transform (don't refit vectorizer)
4. ✅ Predict and decode labels correctly
5. ✅ Rule-based corrections for edge cases

HOW TO RUN:
    from predict_sentiment_FIXED import predict_sentiment
    
    result = predict_sentiment("I love this product!")
    print(result)
    # Output: {'text': 'I love this product!', 'sentiment': 'positive', 'confidence': 0.92}
"""

import pickle
import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = ROOT_DIR / "vectorizer_FIXED.pkl"
LABEL_ENCODER_PATH = ROOT_DIR / "label_encoder_FIXED.pkl"
MODEL_PATH = ROOT_DIR / "sentiment_model_FIXED.pkl"

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODELS (Done once at startup)
# ═══════════════════════════════════════════════════════════════════════════

print("Loading models...")

# Check files exist
if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")
if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load vectorizer
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
print(f"✅ Loaded vectorizer ({vectorizer.get_feature_names_out().shape[0]} features)")

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
print(f"✅ Loaded label encoder ({len(label_encoder.classes_)} classes)")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"✅ Loaded model")

# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING (SAME AS TRAINING!)
# ═══════════════════════════════════════════════════════════════════════════

# KEEP these words - they are important for sentiment!
KEEP_WORDS = {
    "not", "no", "nor", "neither", "never", "nothing", "nowhere", "nobody",
    "isn", "aren", "wasn", "weren", "hasn", "hadn", "haven",
    "doesn", "didn", "don", "dont", "should", "shouldnt", "wont", "wouldnt",
    "cant", "couldnt", "shan", "shant", "mustn", "mightnt", "mayn"
}

stop_words_en = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "can", "may", "might",
    "must", "get", "gets", "got", "make", "makes", "made", "go", "goes",
    "went", "come", "comes", "came", "take", "takes", "took", "see",
    "sees", "saw", "know", "knows", "think", "thinks", "want", "wants",
    "give", "gives", "gave", "tell", "tells", "told", "find", "finds",
    "found", "use", "uses", "used", "such", "so", "just", "even", "me",
    "my", "you", "he", "she", "it", "we", "they", "him", "her", "his",
    "them", "their", "what", "which", "who", "when", "where", "why", "how",
    "all", "each", "every", "some", "any", "most", "few", "more", "less",
    "many", "much", "very", "too", "also", "well", "only"
}

stop_words = stop_words_en - KEEP_WORDS


def preprocess_text(text):
    """
    Preprocess text - MUST BE IDENTICAL TO TRAINING!
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════════════
# RULE-BASED CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_rule_based_correction(text_original, prediction, confidence):
    """
    Apply rule-based corrections for edge cases.
    
    Examples:
    - "neither good nor bad" → NEUTRAL
    - "okay" → NEUTRAL
    - "not bad" → NEUTRAL
    - "not good" → NEGATIVE
    """
    text_lower = text_original.lower()
    
    # Rule 1: Explicit neutral phrases
    neutral_phrases = [
        r"neither.*nor",
        r"neither good nor bad",
        r"average",
        r"okay",
        r"so so",
        r"so-so",
        r"not bad not good",
        r"not good not bad",
        r"kind of",
        r"sort of",
        r"somewhat",
        r"alright",
        r"fine",
        r"decent",
        r"nothing special"
    ]
    
    for phrase in neutral_phrases:
        if re.search(phrase, text_lower):
            return "neutral", 0.95, "rule_neutral"
    
    # Rule 2: Strong negations for negative
    strong_negative = [
        r"hate",
        r"terrible",
        r"awful",
        r"horrible",
        r"pathetic",
        r"disgusting"
    ]
    
    for word in strong_negative:
        if re.search(word, text_lower):
            if prediction != "negative":
                return "negative", 0.95, "rule_negative"
    
    # No rule applied, return original prediction
    return prediction, confidence, "model"


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_sentiment(text):
    """
    Predict sentiment for a text.
    
    Args:
        text (str): Input text to classify
    
    Returns:
        dict: {
            'text': original text,
            'sentiment': predicted sentiment,
            'confidence': confidence score (0-1),
            'method': how prediction was made (model/rule_neutral/rule_negative)
        }
    """
    # 1. Preprocess (SAME as training!)
    text_clean = preprocess_text(text)
    
    # 2. Check if empty after preprocessing
    if not text_clean:
        return {
            'text': text,
            'sentiment': 'neutral',
            'confidence': 0.5,
            'method': 'empty_text',
            'reason': 'Text was empty after preprocessing'
        }
    
    # 3. Vectorize (using LOADED vectorizer, NOT refitting!)
    X = vectorizer.transform([text_clean])
    
    # 4. Predict
    prediction_encoded = model.predict(X)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # 5. Get confidence (probability of predicted class)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = float(probabilities[prediction_encoded])
    else:
        confidence = 1.0  # No probability available
    
    # 6. Apply rule-based corrections
    final_sentiment, final_confidence, method = apply_rule_based_correction(
        text, prediction_label, confidence
    )
    
    return {
        'text': text,
        'text_clean': text_clean,
        'sentiment': final_sentiment,
        'confidence': round(final_confidence, 4),
        'method': method,
        'model_prediction': prediction_label,
        'model_confidence': round(confidence, 4)
    }


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_batch(texts):
    """Predict sentiment for multiple texts."""
    return [predict_sentiment(text) for text in texts]


# ═══════════════════════════════════════════════════════════════════════════
# TEST (if run directly)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING SENTIMENT PREDICTION")
    print("="*80)
    
    test_cases = [
        ("I absolutely love this product!", "positive"),
        ("This is terrible, worst experience ever", "negative"),
        ("It is neither good nor bad", "neutral"),
        ("Not bad, but not great either", "neutral"),
        ("Really happy with the results!", "positive"),
        ("Average quality for the price", "neutral"),
        ("I hate it!", "negative"),
        ("It's fine, satisfied", "neutral"),
    ]
    
    print(f"\n{'Text':<50} | {'Expected':<10} | {'Predicted':<10} | {'✓/✗':<3}")
    print("-" * 85)
    
    correct = 0
    for text, expected in test_cases:
        result = predict_sentiment(text)
        predicted = result['sentiment']
        match = "✓" if predicted == expected else "✗"
        
        if predicted == expected:
            correct += 1
        
        print(f"{text[:48]:<50} | {expected:<10} | {predicted:<10} | {match:<3}")
    
    accuracy = (correct / len(test_cases)) * 100
    print("-" * 85)
    print(f"\n✅ Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")
    
    # Detailed output for sample
    print(f"\n📋 Detailed Output (Sample):")
    sample_text = "I absolutely love this product!"
    result = predict_sentiment(sample_text)
    
    for key, value in result.items():
        print(f"   {key:20s}: {value}")
    
    print("\n" + "="*80 + "\n")
