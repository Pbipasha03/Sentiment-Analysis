# Multi-Class Emotion Classification System

This backend now includes a comprehensive **6-class emotion detection** system alongside the original sentiment analysis.

## Emotion Models

Three machine learning classifiers trained on the emotions dataset:
- **Naive Bayes** (Fastest, 45.45% accuracy)
- **Logistic Regression** (27.27% accuracy)
- **SVM (Linear)** (45.45% accuracy)

### Model Performance (Test Set: 11 samples)

| Model | Accuracy | Macro F1 | Weighted F1 | Train Time | Inference |
|-------|----------|----------|-------------|-----------|-----------|
| Naive Bayes | 0.4545 | 0.3167 | 0.3455 | 0.49ms | 0.05ms |
| Logistic Regression | 0.2727 | 0.2333 | 0.2545 | 5.53ms | 0.05ms |
| SVM (Linear) | 0.4545 | 0.3167 | 0.3455 | 2.69ms | 0.05ms |

**Best Model:** Naive Bayes

### Emotion Classes
- `happy` - Joy, satisfaction, positive emotions
- `sad` - Sadness, disappointment, sorrow
- `angry` - Anger, frustration, rage
- `fear` - Fear, anxiety, worry
- `neutral` - Neutral, mediocre, indifferent
- `surprise` - Surprise, amazement, shock

## Training & Evaluation

### Files Generated

**Model Artifacts:**
- `emotion_vectorizer.pkl` - TF-IDF vectorizer (5000 features, bigrams)
- `emotion_naive_bayes.pkl` - Best performing Naive Bayes model
- `emotion_logistic_regression.pkl` - Logistic Regression model
- `emotion_svm.pkl` - SVM Linear classifier
- `metrics_summary.json` - Comprehensive training metrics

**Visualizations:**
- `confusion_matrices.png` - Confusion matrices for all 3 models
- `accuracy_comparison.html` - Interactive accuracy comparison chart
- `metrics_precision.html` - Per-class precision by emotion
- `metrics_recall.html` - Per-class recall by emotion
- `metrics_f1.html` - Per-class F1-score by emotion

**Dataset:**
- `emotions_dataset.csv` - 52 samples with text and emotion labels (41 train / 11 test)

### Train Your Own Models

```bash
# 1. Update emotions_dataset.csv with more samples (recommended: 500+ samples)
# 2. Run training script
python train_emotion_models.py

# 3. Generate visualizations
python visualize_emotion_models.py
```

## API Endpoints

### Emotion Metrics
```bash
GET /api/emotion/metrics
```

Returns training metrics for all emotion models:
```json
{
  "trained": true,
  "dataset_size": 52,
  "emotion_labels": ["happy", "sad", "angry", "fear", "neutral", "surprise"],
  "train_test_split": "41 / 11",
  "models": [
    {
      "model": "Naive Bayes",
      "accuracy": 0.4545,
      "f1_macro": 0.3167,
      "confusion_matrix": [...],
      "class_report": {...}
    }
  ],
  "best_model": "Naive Bayes"
}
```

### Emotion Analysis
```bash
POST /api/emotion/analyze
Content-Type: application/json

{
  "text": "I'm so happy about this!",
  "model": "naive_bayes"  // optional, defaults to best model
}
```

Response:
```json
{
  "originalText": "I'm so happy about this!",
  "emotion": "happy",
  "confidence": 0.8234,
  "scores": {
    "happy": 0.8234,
    "sad": 0.0123,
    "angry": 0.0234,
    "fear": 0.0156,
    "neutral": 0.1234,
    "surprise": 0.0019
  },
  "model": "naive_bayes",
  "processedText": "happy"
}
```

### Compare Emotion Models
```bash
POST /api/emotion/compare
Content-Type: application/json

{
  "text": "This is heartbreaking news."
}
```

Response:
```json
{
  "text": "This is heartbreaking news.",
  "comparisons": [
    {
      "model": "naive_bayes",
      "emotion": "sad",
      "confidence": 0.7891,
      "scores": {...}
    },
    {
      "model": "logistic_regression",
      "emotion": "sad",
      "confidence": 0.6234,
      "scores": {...}
    },
    {
      "model": "svm",
      "emotion": "sad",
      "confidence": 0.7645,
      "scores": {...}
    }
  ],
  "consensus": "sad",
  "agreement": 1.0,
  "modelCount": 3
}
```

## Improvements for Better Accuracy

The current models achieve ~45% accuracy with only 52 training samples. To improve:

1. **Expand Dataset** (Recommended)
   - Target: 500-1000 balanced samples per emotion class
   - Sources: Kaggle Emotions datasets, Twitter, social media
   - Tools: `pandas`, web scraping, label crowdsourcing

2. **Hyperparameter Tuning**
   ```python
   # In train_emotion_models.py, use GridSearchCV:
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'clf__alpha': [0.1, 1.0, 10.0],  # Naive Bayes smoothing
       'clf__C': [0.01, 0.1, 1.0]       # Logistic Regression regularization
   }
   ```

3. **Feature Engineering**
   - Ngram ranges: Try `(1, 3)` for trigrams
   - Min/max document frequency
   - Lemmatization/stemming
   - Word embeddings (Word2Vec, GloVe)

4. **Model Selection**
   - Try ensemble methods: `RandomForestClassifier`, `GradientBoostingClassifier`
   - Deep learning: LSTM, Transformers (BERT)
   - Transfer learning from pre-trained models

5. **Cross-Validation**
   - Use `cross_val_score()` for robust evaluation
   - Stratified k-fold for imbalanced datasets

## Integration with Frontend

The frontend components can now:

1. **Display emotion metrics** on Dashboard
2. **Run emotion analysis** in Analyze Text page
3. **Compare models** in Compare Emotions page
4. **View confusion matrices** in Model Performance page

Update API client base URL:
```typescript
// artifacts/sentiment-analysis/src/App.tsx
setBaseUrl("http://127.0.0.1:5000");
```

Then use emotion endpoints:
```typescript
const { data: emotionMetrics } = useGetEmotionMetrics();
const emotionMutation = useEmotionAnalyze();
```

## Running the System

```bash
# 1. Start Flask backend
cd backend
python app.py

# 2. Start React frontend (in another terminal)
cd artifacts/sentiment-analysis
npm run dev

# 3. Navigate to http://localhost:4173/emotion/analyze
```

## Next Steps

1. ✅ **Expand the emotions dataset** (current 52 samples → target 500+ samples)
2. ✅ **Fine-tune hyperparameters** using GridSearchCV
3. ✅ **Test on production data** and collect user feedback
4. ⏳ **Implement web UI** for CSV batch emotion analysis
5. ⏳ **Deploy to production** with model versioning
