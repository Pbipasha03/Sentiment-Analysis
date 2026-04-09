# 🎯 Sentiment Analysis ML Pipeline - Complete Guide

## Overview

This is a **complete, production-ready machine learning pipeline** for sentiment analysis using Python, Flask, and Scikit-learn.

**What it does:**
1. ✅ Upload CSV file with text and sentiment labels
2. ✅ Preprocess text (lowercase, remove stopwords, tokenize)
3. ✅ Convert text to numbers using TF-IDF vectorization
4. ✅ Train 3 models: Naive Bayes, Logistic Regression, SVM
5. ✅ Evaluate models with accuracy, precision, recall, F1-score, confusion matrix
6. ✅ Compare models and find the best one
7. ✅ Make predictions on new text
8. ✅ Download results as CSV/JSON reports

---

## 🚀 Quick Start

### Step 1: Start Backend Server

```bash
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer/backend
python3 app_ml_clean.py
```

**Expected output:**
```
🚀 STARTING ML SENTIMENT ANALYSIS API
...
Available Endpoints:
   POST   /api/ml/upload    - Upload CSV and train models
   GET    /api/ml/health    - Health check
   GET    /api/ml/metrics   - Get model metrics
   POST   /api/ml/predict   - Predict sentiment for text
   GET    /api/ml/report/<fmt> - Download report (csv/json)
   GET    /api/ml/status    - Get pipeline status

🔗 Server: http://127.0.0.1:5001
```

### Step 2: Upload CSV & Train Models

**Using Frontend (Recommended):**
1. Open http://localhost:4173/ml-training
2. Click "Choose File"
3. Select your CSV file
4. Click "Train Models"
5. Wait for results to display

**Using cURL (For Testing):**
```bash
curl -X POST "http://127.0.0.1:5001/api/ml/upload" \
  -F "file=@/Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer/backend/sample_dataset_expanded.csv"
```

### Step 3: View Results

Results include:
- **Data Statistics:** Total samples, classes, train/test split, features
- **Model Metrics:** Accuracy, precision, recall, F1-score for each model
- **Confusion Matrices:** Actual vs predicted labels
- **Per-Class Metrics:** Performance breakdown by sentiment class
- **Best Model:** Automatically identified top performer

---

## 📂 File Structure

```
Backend:
├── app_ml_clean.py              ← Main server (IMPROVED - Clean, beginner-friendly)
├── app_ml_complete.py           ← Previous version (still working)
├── sample_dataset_expanded.csv  ← 51-sample test dataset
└── uploads/                     ← Where uploaded CSVs are saved

Frontend:
└── artifacts/sentiment-analysis/
    └── src/pages/ml-training.tsx ← React UI component
```

---

## 📊 CSV Format

Your CSV file must have exactly 2 columns: `text` and `sentiment`

### Example CSV:
```csv
text,sentiment
"I love this product, it's amazing!",Positive
"This is the worst experience ever.",Negative
"It's okay, not too bad.",Neutral
"Absolutely fantastic service!",Positive
"I hate it, very disappointing.",Negative
```

### Requirements:
- ✅ Must be `.csv` file
- ✅ Column names: `text` and `sentiment` (case-sensitive)
- ✅ Minimum 10 rows
- ✅ At least 2 sentiment classes
- ✅ No empty cells (null values removed automatically)

---

## 🔧 Pipeline Steps Explained

### 1. CSV Upload & Validation
```
Input: CSV file
↓
- Check file format (.csv)
- Load file with pandas
- Validate required columns: 'text', 'sentiment'
- Check minimum 10 rows
- Remove rows with null values
↓
Output: List of texts and labels
```

### 2. Text Preprocessing
```
Input: Raw text like "I LOVE this product!!!"
↓
Step 1: Lowercase         → "i love this product!!!"
Step 2: Remove URLs       → No change (no URLs present)
Step 3: Remove special chars → "i love this product"
Step 4: Remove extra spaces  → "i love this product"
Step 5: Remove stopwords    → "love product"
↓
Output: Clean text ready for ML
```

**Why preprocessing?**
- Normalizes text (uppercase/lowercase)
- Removes noise (URLs, special characters)
- Removes common words that don't indicate sentiment
- Creates consistent input for models

### 3. Train/Test Split (Stratified)
```
Input: 51 samples with 3 classes
↓
Split: 80% training (40 samples) | 20% testing (11 samples)
Stratification: Maintains same class proportions
↓
Output: X_train, X_test, y_train, y_test
```

**Why stratification?**
- If 60% of data is "Positive", train and test sets also have 60% "Positive"
- Prevents bias toward one sentiment class

### 4. TF-IDF Vectorization
```
Input: "love product"
↓
TF-IDF converts to numbers:
- "love" → [0.8, 0.2, 0.1, ...]  (high value = important)
- "product" → [0.3, 0.9, 0.2, ...]
↓
Output: Matrix of shape (51 samples, 5000 features)
```

**What is TF-IDF?**
- **TF (Term Frequency):** How often word appears in document
- **IDF (Inverse Document Frequency):** How rare/common word is across all documents
- **High TF-IDF:** Important unique word
- **Low TF-IDF:** Common word (the, is, and)

### 5. Model Training
```
Input: Training data (vectorized text + labels)
↓
Train 3 models:
1. Naive Bayes     - Probability-based, fast
2. Logistic Reg    - Linear classifier, interpretable
3. SVM             - Support Vector Machine, powerful
↓
Output: 3 trained models ready to predict
```

### 6. Model Evaluation
```
Input: Test data (new, unseen samples)
↓
For each model:
- Make predictions on test set
- Compare predictions to true labels
- Compute metrics:
  * Accuracy: % correct predictions
  * Precision: Of predicted positive, how many were correct?
  * Recall: Of actual positive, how many did we find?
  * F1: Balance between precision & recall
  * Confusion Matrix: Detailed breakdown
↓
Output: Evaluation report, best model identified
```

---

## 📈 Understanding Metrics

### Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
- Range: 0 to 1 (or 0% to 100%)
- High accuracy = model predicts correctly most of the time
- Best for: Balanced datasets
```

### Precision
```
Precision = (True Positives) / (True Positives + False Positives)
- "Of predicted Positive, how many were ACTUALLY Positive?"
- High precision = few false alarms
- Best for: When cost of false positive is high
```

### Recall
```
Recall = (True Positives) / (True Positives + False Negatives)
- "Of actual Positive samples, how many did we CATCH?"
- High recall = few missed cases
- Best for: When cost of false negative is high
```

### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Best for: Imbalanced datasets, need balance between metrics
```

### Confusion Matrix
```
                 Predicted
              Pos  Neg  Neu
Actual Pos   [20   2   1]
       Neg   [ 3  18   2]
       Neu   [ 1   0  19]

- Diagonal elements = Correct predictions
- Off-diagonal = Misclassifications
```

---

## 🧪 Testing Examples

### Test 1: Basic CSV Upload

```bash
# Upload and train on sample data
curl -X POST "http://127.0.0.1:5001/api/ml/upload" \
  -F "file=@backend/sample_dataset_expanded.csv"
```

**Expected response:**
```json
{
  "success": true,
  "data_statistics": {
    "total_samples": 51,
    "classes": ["Negative", "Neutral", "Positive"],
    "train_samples": 40,
    "test_samples": 11,
    "features": 17
  },
  "models": {
    "Logistic Regression": {
      "accuracy": 0.3636,
      "precision": 0.1322,
      "recall": 0.3636,
      "f1": 0.1939,
      ...
    }
  },
  "best_model": "Logistic Regression"
}
```

### Test 2: Get Trained Model Metrics

```bash
curl "http://127.0.0.1:5001/api/ml/metrics"
```

### Test 3: Make Prediction

```bash
curl -X POST "http://127.0.0.1:5001/api/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "model": "Logistic Regression"}'
```

**Response:**
```json
{
  "text": "I love this product!",
  "prediction": "Positive",
  "confidence": 0.7123,
  "scores": {
    "Positive": 0.7123,
    "Negative": 0.1534,
    "Neutral": 0.1343
  },
  "model": "Logistic Regression"
}
```

### Test 4: Download Report

```bash
# Download as CSV
curl "http://127.0.0.1:5001/api/ml/report/csv" \
  -o report.csv

# Download as JSON
curl "http://127.0.0.1:5001/api/ml/report/json" \
  -o report.json
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Dataset too small"
**Error:** `Dataset too small: 5 rows. Need at least 10 rows.`

**Solution:**
- Add more training examples to your CSV
- Minimum 10 rows required (preferably 50+)

### Issue 2: "CSV must have 'text' and 'sentiment' columns"
**Error:** `CSV must have 'text' and 'sentiment' columns.`

**Solution:**
- Check column names are exactly `text` and `sentiment`
- Column names are case-sensitive
- Remove other columns (only need 2)

### Issue 3: "Only 1 class found"
**Error:** `Need at least 2 sentiment classes.`

**Solution:**
- CSV must have at least 2 different sentiment values
- Example: "Positive", "Negative", OR "Good", "Bad"
- Make sure you have both in your data

### Issue 4: Train/test split error
**Error:** `test_size = 2 should be greater or equal to the number of classes = 3`

**This is fixed in `app_ml_clean.py`:**
- Uses dynamic test_size calculation
- Works with small datasets
- Ensures each class has enough samples in test set

### Issue 5: "No models trained yet"
**Error:** `No models trained. Please upload CSV first.`

**Solution:**
- Upload a CSV file first to train models
- Go to http://localhost:4173/ml-training and upload file

---

## 🎓 Learning Outcomes

After running this pipeline, you'll understand:

1. **Text Preprocessing**
   - Why we clean and normalize text
   - How stopwords work
   - Impact of preprocessing on model performance

2. **Feature Vectorization**
   - How text is converted to numbers
   - What TF-IDF does and why it's useful
   - Difference between TF, IDF, TF-IDF

3. **Train/Test Split**
   - Why we split data (prevent overfitting)
   - What stratification does
   - Impact of split ratio on model performance

4. **Model Training**
   - How to train multiple models
   - When to use each algorithm
   - Hyperparameter tuning basics

5. **Model Evaluation**
   - How to evaluate classification models
   - Understanding accuracy, precision, recall, F1
   - Reading confusion matrices
   - Comparing multiple models

---

## 🚀 Performance Tips

### For Better Accuracy:

1. **More Data**
   - More training samples = better model
   - Aim for 100+ samples per sentiment class

2. **Better Data Quality**
   - Remove duplicates
   - Fix spelling errors
   - Remove extremely short texts

3. **Balance Classes**
   - Try to have similar number of samples per class
   - Imbalanced data (80% Positive, 20% Negative) causes bias

4. **Hyperparameter Tuning**
   - Adjust TF-IDF parameters (max_features, ngram_range)
   - Experiment with model parameters (C, alpha, kernel)

5. **Feature Engineering**
   - Include ngrams (1-2, 1-3, 2-3 word combinations)
   - Try different max_features (1000, 5000, 10000)

---

## 📚 Code Structure

### Main Components:

**1. `preprocess_text(text)` Function**
- Input: Raw text string
- Output: Cleaned text
- Steps: lowercase → remove URLs → remove special chars → remove stopwords

**2. `SentimentMLPipeline` Class**
- `load_csv()` - Load and validate CSV
- `preprocess()` - Clean all texts
- `vectorize()` - Convert text to numbers
- `train_models()` - Train 3 models
- `evaluate_model()` - Compute metrics
- `full_pipeline()` - Execute complete workflow
- `predict_single()` - Make prediction on new text

**3. Flask Routes**
- `POST /api/ml/upload` - Upload CSV, train models
- `GET /api/ml/health` - Health check
- `GET /api/ml/metrics` - Get metrics
- `POST /api/ml/predict` - Make prediction
- `GET /api/ml/report/<fmt>` - Download report
- `GET /api/ml/status` - Get status

---

## 🔗 API Reference

### POST /api/ml/upload

Upload CSV and train models.

**Request:**
```
POST /api/ml/upload
Content-Type: multipart/form-data
Body: [CSV file]
```

**Response (Success):**
```json
{
  "success": true,
  "data_statistics": {
    "total_samples": 51,
    "classes": ["Negative", "Neutral", "Positive"],
    "train_samples": 40,
    "test_samples": 11,
    "features": 17
  },
  "models": { ... },
  "best_model": "Logistic Regression"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "❌ Dataset too small: 5 rows. Need at least 10 rows."
}
```

---

### GET /api/ml/health

Health check endpoint.

**Response:**
```json
{
  "status": "✓ OK",
  "service": "ML Sentiment Analysis Pipeline",
  "models_trained": true
}
```

---

### POST /api/ml/predict

Predict sentiment for new text.

**Request:**
```json
{
  "text": "I love this product!",
  "model": "Logistic Regression"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "prediction": "Positive",
  "confidence": 0.7123,
  "scores": {
    "Positive": 0.7123,
    "Negative": 0.1534,
    "Neutral": 0.1343
  },
  "model": "Logistic Regression"
}
```

---

### GET /api/ml/metrics

Get all model evaluation metrics.

**Response:**
```json
{
  "models": {
    "Naive Bayes": { ... },
    "Logistic Regression": { ... },
    "SVM": { ... }
  },
  "best_model": "Logistic Regression",
  "summary": {
    "num_models": 3,
    "labels": ["Negative", "Neutral", "Positive"]
  }
}
```

---

### GET /api/ml/report/{format}

Download report as CSV or JSON.

**Request:**
```
GET /api/ml/report/csv
GET /api/ml/report/json
```

**Response:**
- File download with attachment headers

---

## 📊 Comparison: `app_ml_complete.py` vs `app_ml_clean.py`

| Feature | Complete | Clean |
|---------|----------|-------|
| Functionality | ✅ Full | ✅ Full |
| Beginner-friendly | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Documentation | Good | Excellent |
| Comments | Minimal | Comprehensive |
| Error messages | Good | Beginner-friendly |
| Code clarity | Good | Excellent |
| Learning value | Good | Excellent |

**Recommendation:** Use `app_ml_clean.py` for learning and new projects. Both are production-ready.

---

## 🎯 Next Steps

1. **Try the pipeline:**
   - Start backend server
   - Upload sample CSV
   - View results

2. **Use your own data:**
   - Prepare CSV with text and sentiment columns
   - Upload and train on your data
   - Check model performance

3. **Improve models:**
   - Collect more training data
   - Balance sentiment classes
   - Tune hyperparameters

4. **Deploy to production:**
   - Use production WSGI server (gunicorn, waitress)
   - Set debug=False
   - Use environment variables for config
   - Add authentication if needed

---

## 📞 Support

**Common Questions:**

Q: How many samples do I need?
A: Minimum 10, but aim for 50+ per sentiment class for best results.

Q: Why models have low accuracy?
A: Could be small dataset, imbalanced classes, or noisy data. Try preprocessing differently or collecting more data.

Q: Can I use more than 3 models?
A: Yes! Add more model_configs in `train_models()` method.

Q: How do I improve accuracy?
A: More/better data, balance classes, and tune hyperparameters.

---

**Happy Learning! 🎉**
