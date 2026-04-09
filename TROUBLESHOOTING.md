# 🐛 TROUBLESHOOTING GUIDE

## Issue 1: "ModuleNotFoundError: No module named 'sklearn'"

**Error:**
```
ModuleNotFoundError: No module named 'sklearn'
  File "train_sentiment_model_FIXED.py", line 1, in <module>
    from sklearn.preprocessing import LabelEncoder
```

**Solution:**
```bash
# Install scikit-learn
pip install scikit-learn

# Or if using conda
conda install scikit-learn

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

---

## Issue 2: "No such file or directory: 'dataset.csv'"

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset.csv'
```

**Solution:**
```bash
# Check current directory
pwd

# List files
ls -la

# The script looks for dataset.csv in these locations (in order):
# 1. ./dataset.csv (current directory)
# 2. ../dataset.csv (parent directory)
# 3. ../../datasets/dataset.csv

# Make sure your dataset is in one of these locations
# Or create a sample dataset if testing:

python -c "
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'text': [
        'I absolutely love this product!',
        'This is terrible, worst experience',
        'It is neither good nor bad',
        'Amazing quality, highly recommend',
        'Terrible service and poor quality',
        'Average product for the price',
        # ... more samples
    ],
    'label': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'neutral',
        # ... more labels
    ]
}

df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)
print('✅ Sample dataset created: dataset.csv')
"
```

---

## Issue 3: "ValueError: y contains previously unseen labels: ['positive']"

**Error:**
```
ValueError: y contains previously unseen labels: ['positive']
  File "predict_sentiment_FIXED.py", line 45, in <module>
    predictions = model.predict(X_test_vec)
```

**Root Cause:**
```python
# ❌ WRONG: LabelEncoder created during prediction
le = LabelEncoder()
y_test_pred = le.fit_transform(['positive'])  # Unique classes
# This encoder ONLY knows 'positive'
# It doesn't know 'negative' or 'neutral'

# When predicting:
new_le = LabelEncoder()
new_le.fit_transform(['negative', 'neutral', 'positive'])
# Now it knows all 3, but different encoding
```

**Solution:**
```python
# ✅ CORRECT: Load same encoder from training

# In train script:
le = LabelEncoder()
y_train = le.fit_transform(df['label'])
joblib.dump(le, 'label_encoder_FIXED.pkl')  # Save encoder

# In predict script:
le = joblib.load('label_encoder_FIXED.pkl')  # Load same encoder
# le now has EXACT same mapping as training
```

**Prevention:**
```bash
# Make sure these 3 files exist before running prediction:
ls -la *.pkl
# Should see:
# vectorizer_FIXED.pkl
# label_encoder_FIXED.pkl
# sentiment_model_FIXED.pkl

# If missing:
python train_sentiment_model_FIXED.py  # Run training first
```

---

## Issue 4: "Shape mismatch: expected 730 features but got 250"

**Error:**
```
ValueError: X has 250 features, but this estimator is expecting 730 features
  File "predict_sentiment_FIXED.py", line 120, in <module>
    predictions = model.predict(X_test_vec)
```

**Root Cause:**
```python
# ❌ WRONG: Different vectorizer for prediction
# Training used 5000 max_features:
vectorizer_train = TfidfVectorizer(max_features=5000)
X_train = vectorizer_train.fit_transform(texts_train)  # 730 features

# Prediction used 1000 max_features:
vectorizer_pred = TfidfVectorizer(max_features=1000)  # Different!
X_new = vectorizer_pred.fit_transform([new_text])  # 250 features

# Model expects 730, gets 250 → ERROR!
```

**Solution:**
```python
# ✅ CORRECT: Load SAME vectorizer from training

# In train script:
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(texts_train)
print(f"Features: {X_train.shape[1]}")  # 730
joblib.dump(vectorizer, 'vectorizer_FIXED.pkl')

# In predict script:
vectorizer = joblib.load('vectorizer_FIXED.pkl')
X_new = vectorizer.transform([new_text])
print(f"Features: {X_new.shape[1]}")  # 730 (same!)
```

**Debugging:**
```python
# Check vectorizer configuration
import joblib
vectorizer = joblib.load('vectorizer_FIXED.pkl')
print(f"Max features: {vectorizer.max_features}")
print(f"Ngram range: {vectorizer.ngram_range}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
# Should match training configuration
```

---

## Issue 5: "All predictions are 'negative'"

**Error:**
```
✅ Text: "I love this!"
   Predicted: negative (WRONG!)
```

**Root Causes & Solutions:**

### Cause A: Vectorizer refitted during prediction
```python
# ❌ WRONG
vectorizer = joblib.load('vectorizer_FIXED.pkl')
X_new = vectorizer.fit_transform([new_text])  # REFITTING!

# ✅ CORRECT
vectorizer = joblib.load('vectorizer_FIXED.pkl')
X_new = vectorizer.transform([new_text])  # No refit!
```

### Cause B: Different preprocessing
```python
# ❌ WRONG: Different preprocessing
# Training:
texts = [text.lower() for text in texts]  # Just lowercase

# Prediction:
text = text.lower()
text = re.sub(r'[!?]', '', text)  # Remove punctuation
# Different preprocessing → different vectors → wrong prediction!

# ✅ CORRECT: Same preprocessing
# training and prediction use same preprocess_text() function
def preprocess_text(text):
    # ... same steps for both ...
    return cleaned_text

texts = [preprocess_text(t) for t in texts]  # Training
text = preprocess_text(new_text)  # Prediction (same function!)
```

### Cause C: Missing negations in preprocessing
```python
# ❌ WRONG: Removes "not"
text = "This is not good"
# After removing stopwords and "not":
# "this good"
# Model sees: POSITIVE (WRONG!)

# ✅ CORRECT: Keep "not"
text = "This is not good"
# After proper preprocessing:
# "this not good"
# Model sees: NEGATIVE (CORRECT!)
```

**Debugging:**
```python
# Show what's happening at each step
text = "I absolutely love this product!"

print(f"1. Original: {text}")

preprocessed = preprocess_text(text)
print(f"2. After preprocessing: {preprocessed}")

vectorizer = joblib.load('vectorizer_FIXED.pkl')
X_vec = vectorizer.transform([preprocessed])
print(f"3. Vectorized shape: {X_vec.shape}")
print(f"4. Non-zero features: {X_vec.nnz}")

model = joblib.load('sentiment_model_FIXED.pkl')
pred = model.predict(X_vec)[0]
print(f"5. Model prediction: {pred}")

le = joblib.load('label_encoder_FIXED.pkl')
sentiment = le.inverse_transform([pred])[0]
print(f"6. Decoded sentiment: {sentiment}")

# Should see:
# 1. Original: I absolutely love this product!
# 2. After preprocessing: absolutely love product
# 3. Vectorized shape: (1, 730)
# 4. Non-zero features: ~10
# 5. Model prediction: 2
# 6. Decoded sentiment: positive
```

---

## Issue 6: "Empty dataset after preprocessing"

**Error:**
```
ValueError: No samples left after preprocessing
  File "train_sentiment_model_FIXED.py", line 205, in <module>
    X_train_vec = vectorizer.fit_transform(X_train)
```

**Root Cause:**
```python
# ❌ WRONG: Removes too much text
def preprocess_text(text):
    # Remove stopwords (including "not")
    tokens = [w for w in tokens if w not in stopwords]
    
    # For short text like "Not good":
    # "not" removed → empty string
    # "good" might also be removed
    # Result: empty after preprocessing!
```

**Solution:**
```python
# ✅ CORRECT: Keep important words
KEEP_WORDS = {"not", "no", "nor", "neither", "never", ...}
stop_words = get_stop_words() - KEEP_WORDS

def preprocess_text(text):
    tokens = [w for w in tokens if w not in stop_words]
    # Negations preserved
    if not tokens:  # Safety check
        return text  # Return original if empty
    return ' '.join(tokens)

# For "Not good":
# "not" kept, "good" kept
# Result: "not good"
```

**Debugging:**
```python
# Check preprocessing
test_texts = [
    "I love this!",
    "Not good",
    "Neither good nor bad",
    "!!!",  # Only punctuation
]

for text in test_texts:
    cleaned = preprocess_text(text)
    print(f"'{text}' → '{cleaned}'")
    if not cleaned.strip():
        print("  ⚠️ WARNING: Text became empty after preprocessing!")

# Should see:
# 'I love this!' → 'love'
# 'Not good' → 'not good'
# 'Neither good nor bad' → 'neither good nor bad'
# '!!!' → '!!!' (keep non-empty)
```

---

## Issue 7: "90% of predictions are 'neutral'"

**Error:**
```
Text: "I absolutely love this!"
Predicted: neutral (WRONG!)

Predictions: 90% neutral, 5% positive, 5% negative
```

**Root Causes:**

### Cause A: Model untrained
```python
# ❌ WRONG: No training
model = LogisticRegression()
# Never called model.fit()
pred = model.predict(X)  # Predicts random/default class

# ✅ CORRECT: Always train first
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)  # ← Don't skip this!
pred = model.predict(X_test)
```

### Cause B: Imbalanced training data
```python
# ❌ WRONG: Training data mostly neutral
# Training: 5% negative, 90% neutral, 5% positive
# Model learns: "predict neutral most of the time"

# ✅ CORRECT: Balanced training data
# Training: 33% negative, 33% neutral, 34% positive
# Model learns: predict based on actual features
```

### Cause C: All-neutral data during preprocessing
```python
# ❌ WRONG: If preprocessing breaks data
for text in texts:
    processed = preprocess_text(text)
    if not processed.strip():
        processed = "neutral"  # Default to neutral!
```

**Debugging:**
```python
# Check model coefficients
import joblib
model = joblib.load('sentiment_model_FIXED.pkl')

# For LogisticRegression, check intercept
print(f"Intercept: {model.intercept_}")  # Should be close to 0

# Check if model was trained
if hasattr(model, 'coef_'):
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Non-zero coefficients: {(model.coef_ != 0).sum()}")
else:
    print("❌ Model appears untrained!")

# Check predictions on training data
X_train_pred = model.predict(X_train)
unique, counts = np.unique(X_train_pred, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c/len(X_train_pred):.1%}")
# Should see balanced distribution
```

---

## Issue 8: "Accuracy is too low (40%)"

**Error:**
```
Training accuracy: 40%
Test accuracy: 35%
```

**Root Causes:**

### Cause A: Wrong preprocessing
```python
# ❌ WRONG: Removes too much information
text = "I absolutely LOVE this!!!"
→ "love"  # Lost sentiment intensity
```

### Cause B: No class weighting
```python
# ❌ WRONG: Imbalanced learning
model = LogisticRegression()  # No weighting
model.fit(X_train, y_train)  # Biased toward majority class

# ✅ CORRECT
model = LogisticRegression(class_weight='balanced')
```

### Cause C: Too few features
```python
# ❌ WRONG: Very limited features
vectorizer = TfidfVectorizer(max_features=50)  # Too few!

# ✅ CORRECT
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
```

### Cause D: Wrong algorithm for data
```python
# ❌ WRONG: Algorithm doesn't match data
model = SVC(kernel='linear')  # Might not work well

# ✅ CORRECT: Try multiple and pick best
# Logistic Regression: Best for linearly separable
# Naive Bayes: Good for text
# SVM: Better with more data
```

**Solution:**
```python
# Debug step by step
print(f"1. Training samples: {len(X_train)}")
print(f"2. Features: {X_train.shape[1]}")
print(f"3. Class distribution: {np.bincount(y_train)}")
print(f"4. Model: LogisticRegression")
print(f"5. Training accuracy: {model.score(X_train, y_train):.2%}")
print(f"6. Test accuracy: {model.score(X_test, y_test):.2%}")

# If training accuracy low:
# → Issue with data or preprocessing

# If training high, test low:
# → Overfitting (model memorized training data)
```

---

## Issue 9: "Models saved but files are empty"

**Error:**
```
ls -la *.pkl
-rw-r--r--  1 user  staff      0 Apr 10 10:00 vectorizer_FIXED.pkl
-rw-r--r--  1 user  staff      0 Apr 10 10:00 sentiment_model_FIXED.pkl
```

**Root Cause:**
```python
# ❌ WRONG: Files not properly saved
joblib.dump(vectorizer_FIXED.pkl)  # Missing model!

# Or file permission issues
```

**Solution:**
```python
# ✅ CORRECT: Save properly with error checking
try:
    joblib.dump(vectorizer, 'vectorizer_FIXED.pkl')
    print("✅ Vectorizer saved")
except Exception as e:
    print(f"❌ Error saving vectorizer: {e}")

try:
    joblib.dump(label_encoder, 'label_encoder_FIXED.pkl')
    print("✅ Label encoder saved")
except Exception as e:
    print(f"❌ Error saving encoder: {e}")

try:
    joblib.dump(model, 'sentiment_model_FIXED.pkl')
    print("✅ Model saved")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Verify files exist and are not empty
import os
for filename in ['vectorizer_FIXED.pkl', 'label_encoder_FIXED.pkl', 'sentiment_model_FIXED.pkl']:
    size = os.path.getsize(filename)
    if size == 0:
        print(f"❌ {filename} is empty!")
    else:
        print(f"✅ {filename} saved ({size} bytes)")
```

---

## Quick Checklist

Before running scripts:

- [ ] Python 3.7+ installed
- [ ] Required packages installed:
  ```bash
  pip install scikit-learn pandas nltk numpy
  ```
- [ ] Dataset file exists (`dataset.csv`)
- [ ] Dataset has 'text' column and label column ('label' or 'sentiment')
- [ ] Running from correct directory (`pwd` shows correct path)

Before making predictions:

- [ ] Three .pkl files exist: vectorizer, encoder, model
- [ ] Running prediction script from same directory where models were saved
- [ ] Using same preprocessing function
- [ ] Using same vectorizer (don't create new one)

---

## Still Having Issues?

Add debug print statements:

```python
# In predict script, add:
print("=" * 50)
print("DEBUG OUTPUT")
print("=" * 50)

print(f"\n1. Models loading:")
print(f"   Vectorizer shape: {vectorizer.transform(['test']).shape}")
print(f"   Encoder classes: {label_encoder.classes_}")
print(f"   Model type: {type(model).__name__}")

print(f"\n2. Preprocessing:")
test_text = "I love this!"
preprocessed = preprocess_text(test_text)
print(f"   Original: '{test_text}'")
print(f"   Preprocessed: '{preprocessed}'")

print(f"\n3. Vectorization:")
X_vec = vectorizer.transform([preprocessed])
print(f"   Shape: {X_vec.shape}")
print(f"   Non-zero: {X_vec.nnz}")

print(f"\n4. Prediction:")
pred_encoded = model.predict(X_vec)[0]
confidence = model.predict_proba(X_vec)[0].max()
print(f"   Encoded: {pred_encoded}")
print(f"   Confidence: {confidence:.2%}")

print(f"\n5. Decoding:")
pred_label = label_encoder.inverse_transform([pred_encoded])[0]
print(f"   Label: {pred_label}")

print("=" * 50)
```

This will help identify exactly where the issue is!

---

**Need more help? Check the BEFORE_VS_AFTER.md guide for detailed explanations! 🚀**
