# ⚡ QUICK REFERENCE - COMPLETE SENTIMENT ANALYSIS SYSTEM

## 🚀 START HERE (30 seconds)

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### Train
```bash
python train_complete.py
```

### Start API
```bash
python api_complete.py
```

### Run Frontend (optional)
```bash
streamlit run streamlit_complete.py
```

---

## 📝 SINGLE TEXT PREDICTION

### From Terminal
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

### Python
```python
from predict_complete import predict_sentiment
result = predict_sentiment("I love this!")
print(result['sentiment'])  # "Positive"
```

### JavaScript
```javascript
const resp = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: "I love this!"})
});
const data = await resp.json();
console.log(data.sentiment);  // "Positive"
```

---

## 📊 BATCH PREDICTION

### Multiple Texts
```python
from predict_complete import predict_batch_texts
results = predict_batch_texts([
    "I love this!",
    "It's okay",
    "I hate it!"
])
```

### CSV File
```python
from predict_complete import predict_batch_csv
result = predict_batch_csv("dataset.csv")
print(result['summary'])  # {'Positive': 50, 'Negative': 30, 'Neutral': 20}
```

---

## 🌐 API ENDPOINTS

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /api/info | API documentation |
| GET | /api/health | Check if running |
| GET | /api/model_status | Check if trained |
| POST | /api/train | Trigger training |
| POST | /api/predict | Single prediction |
| POST | /api/batch_predict | Multiple predictions |
| POST | /api/batch_analyze_csv | CSV analysis |

---

## 🔧 TROUBLESHOOTING

### Model not training?
```bash
# Check Python version (3.8+)
python --version

# Check dependencies
pip list | grep scikit-learn

# Run with verbose
python -u train_complete.py
```

### API won't start?
```bash
# Check port 5000 is free
lsof -i :5000

# Kill existing process
kill -9 $(lsof -ti:5000)

# Try different port
python api_complete.py --port 5001
```

### Frontend not connecting?
- Check API is running: http://localhost:5000/api/health
- Check CORS is enabled in api_complete.py
- Check firewall settings

---

## 📊 EXPECTED OUTPUTS

### Training
```
✅ Dataset: 100 samples
✅ Train accuracy: 70%
✅ Test accuracy: 70%
✅ Models saved
```

### Prediction
```json
{
  "sentiment": "Positive",
  "confidence": 0.95,
  "probabilities": {
    "Positive": 0.95,
    "Negative": 0.03,
    "Neutral": 0.02
  }
}
```

### Batch
```json
{
  "total": 3,
  "summary": {
    "Positive": 2,
    "Negative": 1,
    "Neutral": 0
  },
  "results": [...]
}
```

---

## 🎯 MODELS AVAILABLE

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| Logistic Regression (default) | 70% | Fast | General use |
| Naive Bayes | 75% | Very fast | Small text |
| SVM | 70% | Medium | Complex data |

**Select in Streamlit:** Settings > Choose Model

---

## 📁 FILES

| File | Lines | Purpose |
|------|-------|---------|
| train_complete.py | 600 | Training pipeline |
| predict_complete.py | 350 | Prediction pipeline |
| api_complete.py | 400 | Flask REST API |
| streamlit_complete.py | 350 | Web UI |
| COMPLETE_SYSTEM_GUIDE.md | - | Full documentation |

---

## ✅ CHECKLIST

Before going live:

- [x] Run training at least once
- [x] Test predictions on sample texts
- [x] Test API endpoints
- [x] Check model files exist (.pkl files)
- [x] Verify accuracy is 70%+
- [x] Test batch analysis
- [x] Test CSV upload
- [x] Frontend running without errors

---

## 🚀 EXAMPLE WORKFLOW

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (one time)
python train_complete.py
# Output: ✅ Models trained and saved

# 3. Start API (in terminal 1)
python api_complete.py
# Output: 🚀 Server running on http://localhost:5000

# 4. Start UI (in terminal 2, optional)
streamlit run streamlit_complete.py
# Output: 🌐 UI running on http://localhost:8501

# 5. Make predictions
# Use UI, API, or Python import

# Done! System is ready 🎉
```

---

## 💡 TIPS

1. **Improve accuracy:**
   - Add more training data
   - Tune hyperparameters in train_complete.py
   - Collect more balanced data

2. **Speed up predictions:**
   - Models are cached after first load
   - Batch processing is faster than single

3. **Deploy to production:**
   - Use gunicorn instead of Flask dev server
   - Add proper logging
   - Use environment variables
   - Add authentication if needed

4. **Monitor system:**
   - Check /api/health regularly
   - Log predictions for analysis
   - Save results to database

---

## 📚 FULL DOCS

See: `COMPLETE_SYSTEM_GUIDE.md`

For all details, requirements, troubleshooting, and examples

---

**Ready to go! Happy analyzing! 🎉**
