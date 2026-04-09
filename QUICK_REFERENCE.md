# 🚀 ML Sentiment Analysis - Quick Reference

## ✅ What You Now Have

### 1. **Two Production-Ready Backend Servers**
- `app_ml_clean.py` ← **RECOMMENDED** (Clean, beginner-friendly, extensive comments)
- `app_ml_complete.py` (Original, fully functional)

### 2. **Complete ML Pipeline**
```
CSV Upload → Validate → Preprocess → Vectorize → Train → Evaluate → Results
```

### 3. **Interactive React Frontend**
- File upload UI
- Real-time training progress
- Model comparison tables
- Performance metrics display
- Prediction tester
- Report downloads

### 4. **3 Trained Models**
- Naive Bayes - Fast, probabilistic
- Logistic Regression - Linear, interpretable  
- SVM - Powerful, non-linear patterns

### 5. **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance metrics
- Model ranking/comparison

---

## 🎯 How to Use

### Start Backend
```bash
python3 /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer/backend/app_ml_clean.py
```

### Open Frontend
```
http://localhost:4173/ml-training
```

### Upload Your CSV
1. Click "Choose File"
2. Select CSV with 'text' and 'sentiment' columns
3. Click "Train Models"
4. View results immediately

---

## 📊 CSV Format
```
text,sentiment
"I love this!",Positive
"Terrible experience",Negative
"It's okay",Neutral
```
- Minimum: 10 rows, 2 columns, 2+ sentiment classes
- Recommended: 50+ rows, balanced classes

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | /api/ml/upload | Upload CSV, train models |
| GET | /api/ml/health | Health check |
| GET | /api/ml/metrics | Get all model metrics |
| POST | /api/ml/predict | Predict sentiment for text |
| GET | /api/ml/report/csv | Download CSV report |
| GET | /api/ml/report/json | Download JSON report |
| GET | /api/ml/status | Get pipeline status |

---

## 🧪 Quick Test

```bash
# Test prediction
curl -X POST "http://127.0.0.1:5001/api/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!", "model": "Logistic Regression"}'
```

---

## 📁 File Locations
- Backend: `/backend/app_ml_clean.py`
- Frontend: `/artifacts/sentiment-analysis/src/pages/ml-training.tsx`
- Sample Data: `/backend/sample_dataset_expanded.csv`
- Full Guide: `/ML_PIPELINE_GUIDE.md`

---

## ⚡ Key Improvements in `app_ml_clean.py`

✅ Extensive inline comments explaining every step
✅ Comprehensive docstrings for all functions
✅ Beginner-friendly variable names
✅ Clear error messages for troubleshooting
✅ ASCII art headers for visual clarity
✅ Step-by-step pipeline logging
✅ Dynamic test_size for small datasets
✅ Better data validation messages
✅ Example outputs in docstrings

---

## 🎓 What You'll Learn

**Text Processing:**
- How to clean and normalize text
- Stopwords and why they matter
- Text tokenization

**Feature Engineering:**
- TF-IDF vectorization
- N-gram extraction
- Feature importance

**Machine Learning:**
- Model training and evaluation
- Cross-validation and stratified splitting
- Hyperparameter tuning
- Model comparison

**Metrics & Evaluation:**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance analysis

---

## 🚀 Next Steps

1. **Test with sample data** (already provided)
   ```bash
   Upload: /backend/sample_dataset_expanded.csv
   ```

2. **Use your own data**
   - Prepare CSV with 'text' and 'sentiment' columns
   - Aim for 50+ samples per class
   - Balance sentiment distribution

3. **Improve model**
   - Collect more quality data
   - Balance sentiment classes
   - Experiment with preprocessing
   - Tune hyperparameters

4. **Deploy to production**
   - Replace Flask dev server with Gunicorn/Waitress
   - Set environment variables for secrets
   - Add authentication/authorization
   - Add rate limiting
   - Use HTTPS

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run: `pip install flask flask-cors scikit-learn pandas nltk` |
| Port already in use | Change port in code or kill process: `pkill -f app_ml_clean.py` |
| CORS errors | CORS already enabled, check frontend URL |
| CSV validation error | Use exact column names 'text' and 'sentiment' |
| Low accuracy | Add more training data, balance classes |
| Slow training | Reduce max_features in TF-IDF config |

---

## 📚 Resources

- **scikit-learn docs:** https://scikit-learn.org/stable/
- **Pandas docs:** https://pandas.pydata.org/
- **Flask docs:** https://flask.palletsprojects.com/

---

## 💡 Tips

✨ Always preprocess text consistently (train & test)
✨ Use stratified splitting for imbalanced data
✨ Validate data BEFORE training
✨ Start with simpler models (NB) before complex ones (SVM)
✨ Use confusion matrix to understand specific misclassifications
✨ Keep test data completely separate until evaluation

---

**Ready to go! 🎉**

Start the server and upload your CSV to begin sentiment analysis.
