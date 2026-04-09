# 🚀 Batch Analysis - Complete Fix Guide

## What Was Fixed

✅ **Backend:** Added `/api/sentiment/analyze-batch` endpoint  
✅ **Frontend:** Improved CSV parsing and error handling  
✅ **File Upload:** Better validation and user feedback  
✅ **Results Display:** Fixed to show table with predictions  
✅ **Error Handling:** User-friendly error messages  

---

## System Requirements

### Backend Running
```bash
python3 /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer/backend/app.py
# Expected: Running on http://127.0.0.1:5000
```

### Frontend Running
```bash
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer
npm exec -- pnpm --filter @workspace/sentiment-analysis dev
# Expected: Running on http://localhost:4173
```

---

## How to Test Batch Analysis

### Step 1: Train Models (if not already trained)

1. Go to http://localhost:4173/ml-training
2. Upload a CSV file with text and sentiment
3. Click "Train Models"
4. Wait for models to train (2-5 seconds)

### Step 2: Upload Batch CSV

1. Go to http://localhost:4173/batch
2. Select a model from dropdown (Naive Bayes, Logistic Regression, or SVM)
3. Click "Upload CSV"
4. Select a CSV file with text data

### Step 3: View Results

Results will show:
- **Results Table:** Text → Sentiment (Positive/Negative/Neutral) → Confidence
- **Distribution Chart:** Pie chart showing count breakdown
- **Processing Time:** How long it took to analyze all texts

---

## CSV File Format

Your CSV file should have:
- **First column:** Text data (required)
- **First row:** Can be header or data

### Example 1: With Header
```csv
text
I love this product!
This is terrible
It's okay
Amazing service
```

### Example 2: Without Header
```csv
I love this product!
This is terrible
It's okay
Amazing service
```

### Example 3: Comma-Separated with Quoted Strings
```csv
text,sentiment
"I love this!",positive
"Very bad experience",negative
```

---

## API Endpoint

### Request
```
POST /api/sentiment/analyze-batch
Content-Type: application/json

{
  "data": {
    "texts": [
      "I love this!",
      "This is terrible",
      "It's okay"
    ],
    "model": "naive_bayes"  // or "logistic_regression", "svm"
  }
}
```

### Response Success
```json
{
  "results": [
    {
      "text": "I love this!",
      "label": "positive",
      "confidence": 0.95,
      "scores": {
        "positive": 0.95,
        "negative": 0.03,
        "neutral": 0.02
      }
    },
    ...
  ],
  "summary": {
    "total": 3,
    "positive": 1,
    "negative": 1,
    "neutral": 1
  },
  "processingTimeMs": 123,
  "model": "naive_bayes"
}
```

### Response Error
```json
{
  "error": "Models not trained. Please train models first."
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Train Models First" button | Go to /ml-training and train models first |
| "No text data found" error | Make sure first column contains text |
| "File too large" error | Limit CSV to < 10MB or < 1000 rows |
| "Failed to fetch" error | Check backend is running on http://127.0.0.1:5000 |
| No results displayed | Check browser console for errors |

---

## Code Changes Summary

### Backend (`app.py`)
**New Endpoint:** `POST /api/sentiment/analyze-batch`
- Location: Line 350-424
- Validates input
- Processes each text
- Returns formatted results with summary

**Key Features:**
- Batch processing (up to 1000 texts per request)
- Error handling for individual texts
- Processing time tracking
- Summary statistics

### Frontend (`batch.tsx`)
**Improved CSV Parsing:**
- Better quoted string handling
- Header detection
- File validation (type, size)
- Clear error messages

**Better Error Handling:**
- File upload validation
- User-friendly error alerts
- Loading states
- Success feedback

**Display Improvements:**
- Results table with text preview
- Sentiment badges with color coding
- Confidence scores as percentages
- Pie chart distribution
- Processing time display

---

## Testing Checklist

- [ ] Backend running on 5000
- [ ] Frontend running on 4173
- [ ] Models trained (green checkmark)
- [ ] Can select model from dropdown
- [ ] Can click upload button
- [ ] Can select CSV file
- [ ] CSV parses without errors
- [ ] Results display in table
- [ ] Pie chart shows distribution
- [ ] Confidence scores visible
- [ ] Processing time displayed

---

## Sample CSV for Testing

**small-batch.csv:**
```csv
text
I absolutely love this!
This is the worst thing ever
It's pretty good
Amazing quality
Terrible experience
Not bad
Excellent service
Very disappointing
```

**Save as:** `/tmp/small-batch.csv`

Then upload to http://localhost:4173/batch

---

## Performance Expected

| Texts | Time |
|-------|------|
| 10 | < 500ms |
| 50 | < 1s |
| 100 | 1-2s |
| 500 | 5-10s |
| 1000 | 10-20s |

---

## Browser Debugging

**Open DevTools (F12):**

1. **Console tab:** Check for JavaScript errors
2. **Network tab:** Check `/api/sentiment/analyze-batch` request
3. **Application tab:** Check stored state

**Common issues:**
- ❌ CORS error → Check backend CORS setup
- ❌ 404 error → Endpoint not found
- ❌ 500 error → Backend error (check terminal)
- ❌ Timeout → Dataset too large

---

## Next Steps

1. **Test batch analysis** with various CSV files
2. **Monitor performance** with large batches
3. **Collect feedback** on UI/UX
4. **Fine-tune models** based on accuracy
5. **Deploy to production** when ready

---

**Ready to analyze! 🚀**
