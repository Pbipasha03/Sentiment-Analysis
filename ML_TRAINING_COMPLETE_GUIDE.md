# 🚀 Complete ML Training Pipeline - Setup & Usage Guide

## Overview

This is a complete machine learning sentiment analysis system with:
- ✅ CSV file upload
- ✅ Automatic text preprocessing (tokenization, stopword removal)
- ✅ TF-IDF vectorization
- ✅ Training 3 models: Naive Bayes, Logistic Regression, SVM
- ✅ Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
- ✅ Model comparison and ranking
- ✅ Single text prediction
- ✅ Report generation (CSV/JSON download)
- ✅ Complete error handling

## 📋 Architecture

```
Frontend (React Vite)              Backend (Flask)
├── ML Training UI                 ├── CSV Upload Handler
├── File Upload Component          ├── Text Preprocessing
├── Training Progress              ├── TF-IDF Vectorization
├── Metrics Display                ├── Model Training Pipeline
├── Model Comparison Table         ├── Model Evaluation
├── Per-Class Metrics              ├── Metrics Computation
├── Single Text Prediction        ├── Report Generation
└── Download Report               └── Prediction Endpoint

http://localhost:4173             http://127.0.0.1:5001
```

## 🔧 Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies (if not already installed)
pip3 install flask flask-cors scikit-learn pandas numpy nltk

# Run the ML training server
python3 app_ml_complete.py
```

**Expected Output:**
```
🚀 Starting ML Sentiment Analysis API...
📊 Endpoints:
   POST /api/ml/upload       - Upload CSV and train models
   GET  /api/ml/metrics      - Get model evaluation metrics
   POST /api/ml/predict      - Predict sentiment for text
   GET  /api/ml/report/<fmt> - Download report (csv/json)
   GET  /api/ml/status       - Get pipeline status

 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5001
```

### 2. Frontend Setup (in another terminal)

```bash
# Navigate to frontend directory
cd artifacts/sentiment-analysis

# Run development server
PORT=4173 BASE_PATH=/ npm run dev
```

**Expected Output:**
```
  VITE v7.3.1  ready in XXX ms

  ➜  Local:   http://localhost:4173/
  ➜  Network: http://192.168.x.x:4173/
  ➜  press h + enter to show help
```

## 📊 How to Use

### Step 1: Navigate to ML Training Page

Open browser: **http://localhost:4173**

Click on **"ML Training"** in the sidebar navigation

### Step 2: Upload CSV File

1. Prepare a CSV file with columns:
   - `text` - The text to analyze
   - `sentiment` - The target label (e.g., positive, negative, neutral)

   **Example:**
   ```
   text,sentiment
   "I love this product!",positive
   "This is terrible",negative
   "It's okay",neutral
   ```

2. Click the upload area and select your CSV file
3. Click **"Train Models"** button

### Step 3: Monitor Training

The system will:
1. ✓ Load and validate CSV
2. ✓ Preprocess texts (lowercase, remove stopwords, tokenize)
3. ✓ Split data 80/20 (train/test)
4. ✓ Vectorize with TF-IDF
5. ✓ Train 3 models (NB, LR, SVM)
6. ✓ Evaluate each model
7. ✓ Display results

### Step 4: Review Results

#### Model Comparison Table
Shows for each model:
- **Accuracy** - Overall correctness percentage
- **Precision** - How many predicted positives were actually positive
- **Recall** - How many actual positives were correctly identified
- **F1-Score** - Harmonic mean of precision and recall

#### Per-Class Metrics
Detailed metrics for each sentiment class:
- Precision, Recall, F1-score per class
- Support (number of test samples for each class)

#### Best Model
Automatically highlighted with 🏆 badge

### Step 5: Test Predictions

1. Select a model from the dropdown
2. Enter text to analyze
3. Click **"Analyze Sentiment"**
4. View prediction and confidence scores

### Step 6: Download Report

- 📊 **CSV Report** - Opens in Excel/Sheets
- 📋 **JSON Report** - Raw metrics in JSON format

## 📁 What Gets Saved

```
backend/
├── uploads/
│   ├── your_file.csv              - Your uploaded file
│   ├── sentiment_analysis_report_YYYYMMDD_HHMMSS.csv
│   └── sentiment_analysis_report_YYYYMMDD_HHMMSS.json
└── app_ml_complete.py             - Main API server
```

## 🧪 Test with Sample Data

A sample CSV file is provided:

```bash
# Copy it
cp backend/sample_training_data.csv backend/my_test_data.csv
```

Then upload through the UI.

## 📱 API Endpoints Reference

### Health Check
```bash
GET http://127.0.0.1:5001/api/ml/health
```

### Upload & Train
```bash
POST http://127.0.0.1:5001/api/ml/upload
Content-Type: multipart/form-data

file: <your_csv_file>
```

**Response:**
```json
{
  "success": true,
  "data_statistics": {
    "total_samples": 45,
    "classes": ["positive", "negative", "neutral"],
    "train_samples": 36,
    "test_samples": 9,
    "features": 142
  },
  "models": {
    "Naive Bayes": {
      "accuracy": 0.8889,
      "precision": 0.8889,
      "recall": 0.8889,
      "f1": 0.8889,
      "confusion_matrix": [[3,0,0],[0,3,0],[0,0,3]],
      "per_class": {...}
    },
    ...
  },
  "best_model": "Naive Bayes"
}
```

### Get Metrics
```bash
GET http://127.0.0.1:5001/api/ml/metrics
```

### Make Prediction
```bash
POST http://127.0.0.1:5001/api/ml/predict
Content-Type: application/json

{
  "text": "I love this product!",
  "model": "Naive Bayes"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "prediction": "positive",
  "scores": {
    "positive": 0.9234,
    "negative": 0.0515,
    "neutral": 0.0251
  },
  "model": "Naive Bayes",
  "confidence": 0.9234
}
```

### Download Report
```bash
GET http://127.0.0.1:5001/api/ml/report/csv   # CSV format
GET http://127.0.0.1:5001/api/ml/report/json  # JSON format
```

## 🐛 Troubleshooting

### "Address already in use" Error

**Problem:** Port 5001 is already in use

**Solution:**
```bash
# Kill the process using port 5001
lsof -ti:5001 | xargs kill -9

# Then restart
python3 app_ml_complete.py
```

### CSV Upload Fails

**Problem:** "Only CSV files are supported"

**Solution:**
- Ensure file has `.csv` extension
- Check file is valid CSV format
- Open in Excel/Sheets and re-save

### "CSV must have 'text' and 'sentiment' columns"

**Problem:** Wrong column names

**Solution:**
- Column names must be exactly `text` and `sentiment` (lowercase)
- Check for extra spaces in column names

### Models Show Low Accuracy

**Common Causes:**
1. **Too few samples** - Try with 100+ samples minimum (50 per class)
2. **Imbalanced classes** - Ensure roughly equal samples per class
3. **Dirty data** - Remove duplicates, fix typos
4. **Class overlap** - Some texts may be ambiguous

**Solution:**
- Expand dataset
- Balance class distribution
- Add more distinct examples

## 🚀 Best Practices

### CSV File Preparation

✅ **DO:**
- Use clear, descriptive text
- Ensure consistent labeling
- Remove duplicates
- Balance classes (equal samples per class)
- Use UTF-8 encoding

❌ **DON'T:**
- Mix languages
- Use inconsistent labels (e.g., "Positive" vs "positive")
- Include HTML/special characters
- Leave empty text or labels

### Expected Performance

| Dataset Size | Expected Accuracy |
|--------------|-------------------|
| 30-50 samples | 60-75% |
| 100-150 samples | 75-85% |
| 500+ samples | 85-95% |

## 📈 Model Comparison

### Naive Bayes
- ✅ Fast training (< 1ms)
- ✅ Works well with small datasets
- ✅ Good baseline
- ❌ Assumes feature independence

### Logistic Regression
- ✅ Good generalization
- ✅ Interpretable
- ✅ Linear decision boundaries
- ❌ Slower than NB

### SVM (Support Vector Machine)
- ✅ Best accuracy on complex data
- ✅ Handles high dimensions well
- ✅ Non-linear boundaries
- ❌ Slower training & prediction

## 💡 Tips for Better Results

1. **Use balanced data** - Equal samples per class
2. **Clean your data** - Remove noise, fix spellings
3. **Use domain-specific data** - Relevant to your use case
4. **Add more samples** - More data = better models
5. **Explore features** - TF-IDF settings can be tuned
6. **Cross-validate** - Test on different data splits

## 📚 Algorithm Details

### Text Preprocessing
```
Raw Text
  ↓ Lowercase
  ↓ Remove URLs, mentions, hashtags
  ↓ Remove punctuation & special chars
  ↓ Tokenize (split into words)
  ↓ Remove stopwords (the, a, is, etc.)
  ↓ Filter short words (< 3 chars)
Clean Text
```

### TF-IDF Vectorization
- **Max Features:** 5000 most important words
- **N-grams:** 1-2 (single words + word pairs)
- **Min DF:** Word must appear in ≥ 2 documents
- **Max DF:** Word appears in ≤ 80% of documents

### Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | Correct / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted X, how many were right |
| **Recall** | TP / (TP + FN) | Of actual X, how many we found |
| **F1-Score** | 2 × (P × R) / (P + R) | Balanced metric |

## 🎯 Next Steps

1. ✅ Run the system with sample data
2. ✅ Prepare your own CSV file
3. ✅ Train models on your data
4. ✅ Compare model performance
5. ✅ Make predictions on new texts
6. ✅ Download and review reports
7. ⏳ Expand dataset for better accuracy
8. ⏳ Experiment with hyperparameters
9. ⏳ Deploy to production

---

**Questions?** Check the API endpoints or review the backend code in `app_ml_complete.py`
