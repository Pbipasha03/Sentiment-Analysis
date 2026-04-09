"""
improved_prediction.py
======================
Improved prediction module with:
- Better preprocessing (keeping negation words)
- N-gram support for phrase understanding
- Confidence-based neutral classification
- Rule-based neutral detection
- Proper probability handling

Usage in app.py:
    from improved_prediction import ImprovedPredictor
    
    predictor = ImprovedPredictor()
    result = predictor.predict("I love this product!")
"""

import pickle
import re
from pathlib import Path
from typing import Tuple, Dict

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# Get improved stopwords (keeping negation words)
def get_improved_stopwords():
    """Get stopwords but KEEP negation words."""
    stop_words = set(stopwords.words("english"))
    keep_words = {
        "not", "no", "nor", "neither", "never", "nothing", "nowhere",
        "nobody", "isn", "aren", "wasn", "weren", "hasn", "hadn",
        "haven", "doesnt", "didnt", "dont", "should", "shouldnt",
        "wont", "wouldnt", "cant", "couldnt", "shan", "shant"
    }
    return stop_words - keep_words


STOP_WORDS_IMPROVED = get_improved_stopwords()
LABELS = ["negative", "neutral", "positive"]
ROOT_DIR = Path(__file__).resolve().parent


def preprocess_improved(text: str) -> str:
    """Enhanced preprocessing preserving negation and important phrases."""
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
    Returns (is_neutral, confidence)
    """
    text_lower = text.lower()
    neutral_phrases = [
        "neither.*nor", "neither good nor bad", "average", "okay",
        "so so", "not bad not good", "just okay", "moderate",
        "alright", "fine", "decent", "nothing special", "not special"
    ]
    
    for phrase in neutral_phrases:
        if re.search(phrase, text_lower):
            return True, 0.95
    
    return False, 0.0


class ImprovedPredictor:
    """
    Improved sentiment predictor with:
    - Better preprocessing
    - N-gram support
    - Confidence thresholds
    - Rule-based neutral detection
    """
    
    def __init__(
        self,
        vectorizer_path: str = "vectorizer_improved.pkl",
        model_path: str = "sentiment_lr_improved.pkl",
        confidence_threshold: float = 0.4
    ):
        """
        Initialize predictor.
        
        Args:
            vectorizer_path: Path to TF-IDF vectorizer
            model_path: Path to model (NB, LR, or SVM)
            confidence_threshold: Min confidence before classifying as neutral
        """
        self.confidence_threshold = confidence_threshold
        self.vectorizer = None
        self.model = None
        self.labels = LABELS
        
        # Try to load models from ROOT_DIR
        vec_path = Path(ROOT_DIR) / vectorizer_path
        mdl_path = Path(ROOT_DIR) / model_path
        
        if vec_path.exists() and mdl_path.exists():
            with open(vec_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(mdl_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Model files not found:\n"
                f"  Vectorizer: {vec_path}\n"
                f"  Model: {mdl_path}\n"
                f"  Run train_improved_models.py first"
            )
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment with improved logic.
        
        Returns:
            {
                "text": original text,
                "label": "positive" | "neutral" | "negative",
                "confidence": 0.0-1.0,
                "probabilities": {"negative": ..., "neutral": ..., "positive": ...},
                "method": "model" | "threshold" | "rule" | "confidence",
                "cleaned_text": preprocessed text
            }
        """
        # Rule-based neutral detection
        is_neutral_rule, conf_rule = rule_based_neutral_detection(text)
        
        # Preprocess
        clean_text = preprocess_improved(text)
        
        if not clean_text:
            return {
                "text": text,
                "label": "neutral",
                "confidence": 0.5,
                "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
                "method": "empty_text",
                "cleaned_text": clean_text
            }
        
        # Vectorize
        X = self.vectorizer.transform([clean_text])
        
        # Get prediction
        label = self.model.predict(X)[0]
        
        # Get probabilities if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            label_idx = list(self.model.classes_).index(label)
            confidence = float(proba[label_idx])
            
            # Map probabilities to standard labels
            probabilities = {}
            for i, cls_label in enumerate(self.model.classes_):
                normalization_label = cls_label  
                probabilities[normalization_label] = float(proba[i])
        else:
            confidence = 1.0
            probabilities = {self.labels[0]: 0.0, label: 1.0, self.labels[-1]: 0.0}
        
        # Apply decision logic
        # 1. Check rule-based neutral detection first
        if is_neutral_rule:
            return {
                "text": text,
                "label": "neutral",
                "confidence": conf_rule,
                "probabilities": probabilities,
                "method": "rule_based",
                "cleaned_text": clean_text
            }
        
        # 2. Check confidence threshold
        if confidence < self.confidence_threshold:
            # Low confidence → classify as neutral
            neutral_prob = 1.0 - (confidence / self.confidence_threshold)
            probs_adjusted = probabilities.copy()
            probs_adjusted["neutral"] = max(probs_adjusted.get("neutral", 0.0), neutral_prob)
            
            return {
                "text": text,
                "label": "neutral",
                "confidence": neutral_prob,
                "probabilities": probs_adjusted,
                "method": "confidence_threshold",
                "cleaned_text": clean_text,
                "original_confidence": confidence,
                "original_label": label
            }
        
        # 3. Normal model prediction
        return {
            "text": text,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "method": "model",
            "cleaned_text": clean_text
        }
    
    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for batch of texts."""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # Test the improved predictor
    predictor = ImprovedPredictor()
    
    test_texts = [
        "I love this product!",
        "This is terrible",
        "It is neither good nor bad",
        "Not bad, but not great",
        "Average quality",
    ]
    
    print("\n" + "="*70)
    print("Testing Improved Predictor")
    print("="*70)
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n📝 Text: {text}")
        print(f"   Label: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"   Method: {result['method']}")
        print(f"   Probs: {result['probabilities']}")
