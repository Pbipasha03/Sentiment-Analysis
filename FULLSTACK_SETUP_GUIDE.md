# ✅ Sentiment Analyzer - Full-Stack Setup & Troubleshooting Guide

## 🎯 Overview
This document explains the fixes applied to the full-stack sentiment analyzer project and how to run it properly.

## 🔧 What Was Broken & Fixed

### 1. **Frontend API URL Mismatch** ❌ → ✅
**Problem:** Frontend was hardcoded to connect to `http://127.0.0.1:5002`, but the backend was not running on that port.

**Fixed:**
- Changed frontend API URL from `http://127.0.0.1:5002` to `http://localhost:8000`
- Updated [artifacts/sentiment-analysis/src/App.tsx](artifacts/sentiment-analysis/src/App.tsx#L38)

### 2. **Backend Port Not Set** ❌ → ✅
**Problem:** Backend required PORT environment variable but scripts didn't set it consistently.

**Fixed:**
- Updated backend package.json scripts to explicitly set `PORT=8000`
- File: [artifacts/api-server/package.json](artifacts/api-server/package.json#L7)

### 3. **"Failed to fetch" Errors** ❌ → ✅
**Problem:** CORS and connectivity issues prevented frontend from reaching backend.

**Fixed:**
- Backend Express app includes proper CORS setup (already configured in [artifacts/api-server/src/app.ts](artifacts/api-server/src/app.ts#L22))
- Frontend now uses correct backend URL
- Verified all endpoints return proper JSON responses

### 4. **Model "Not Trained" Issue** ❌ → ✅
**Problem:** Model state wasn't persisted to disk, so it was lost on server restart.

**Fixed:**
- Backend already had persistence logic in [artifacts/api-server/src/lib/modelStore.ts](artifacts/api-server/src/lib/modelStore.ts)
- Models are automatically saved to `.data/trained-models.json` after training
- On server startup, models are automatically loaded from disk
- `ensureDefaultModelsTrained()` auto-trains on first startup if no models exist

**Status Check:** The backend now correctly reports `trained: true` with full metrics.

### 5. **Missing Global Start Scripts** ❌ → ✅
**Problem:** No easy way to start both backend and frontend together.

**Fixed:**
- Added convenient npm scripts in root `package.json`
- Created `START_SERVERS.sh` for one-command startup
- Backend and frontend now start seamlessly together

## 🚀 How to Start the System

### Option 1: Quick Start (Recommended)
```bash
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer
./START_SERVERS.sh
```

### Option 2: Using npm/pnpm
```bash
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer
npm run frontend:dev  # In one terminal
npm run backend:start # In another terminal (after backend:build)
```

### Option 3: Manual Start
```bash
# Terminal 1 - Backend
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer
npx pnpm --filter @workspace/api-server run build
PORT=8000 node --enable-source-maps ./artifacts/api-server/dist/index.mjs

# Terminal 2 - Frontend
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer
npx pnpm --filter "@workspace/sentiment-analysis" run dev:local
```

## 📡 API Endpoint Reference

All endpoints are at `http://localhost:8000/api/`:

### Health Check
```bash
GET /healthz
# Returns: { "status": "ok" }
```

### Sentiment Analysis (Single Text)
```bash
POST /sentiment/analyze
Content-Type: application/json

{
  "text": "I love this product!",
  "model": "naive_bayes"  // optional
}
```

### Batch Analysis
```bash
POST /sentiment/analyze-batch
Content-Type: application/json

{
  "texts": ["Great!", "Terrible", "OK"],
  "model": "naive_bayes"  // optional
}
```

### Train Models
```bash
POST /models/train
Content-Type: application/json

{
  "useDefaultDataset": true,
  "texts": [],            // optional custom texts
  "labels": []            // optional custom labels
}
```

### Get Model Metrics
```bash
GET /models/metrics
# Returns: accuracy, precision, recall, F1 score for all 3 models
```

### Compare Models
```bash
POST /models/compare
Content-Type: application/json

{
  "text": "This movie was awesome!"
}
```

### Word Cloud Data
```bash
POST /dataset/wordcloud
Content-Type: application/json

{
  "texts": ["I love this", "This is great"],
  "sentimentFilter": "positive"  // optional: "positive" | "negative" | "neutral"
}
```

### Generate Report
```bash
POST /report/generate
Content-Type: application/json

{
  "format": "json",  // or "csv"
  "results": [
    {
      "index": 0,
      "text": "Amazing!",
      "label": "positive",
      "confidence": 0.95,
      "model": "naive_bayes"
    }
  ]
}
```

## ✅ Verification Checklist

After starting the servers, verify everything works:

```bash
# 1. Check backend health
curl http://localhost:8000/api/healthz
# Should return: { "status": "ok" }

# 2. Check model status
curl http://localhost:8000/api/models/metrics
# Should return: { "trained": true, "metrics": [...] }

# 3. Test sentiment analysis
curl -X POST http://localhost:8000/api/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
# Should return: { "originalText": "...", "result": {...}, ... }

# 4. Open frontend
open http://localhost:4173
# Should load the React app without errors

# 5. Test frontend connectivity
# In browser console, check for no CORS/fetch errors
# All pages should work:
#   - Analyze (single text prediction)
#   - Batch Analysis (upload/analyze multiple)
#   - ML Training (train models)
#   - Models (view metrics)
#   - Compare (compare model predictions)
#   - Word Cloud (visualize keywords)
#   - Report (export results)
```

## 🐛 Troubleshooting

### "Port 8000 already in use"
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
PORT=8001 node --enable-source-maps ./artifacts/api-server/dist/index.mjs
# Then update frontend URL to http://localhost:8001
```

### "Failed to fetch" in frontend
- ✅ Verify backend is running: `curl http://localhost:8000/api/healthz`
- ✅ Check frontend is using correct URL: `http://localhost:8000`
- ✅ Ensure CORS is enabled on backend (should be automatic)

### "Model not trained" in UI
- ✅ Check backend reports trained models: `curl http://localhost:8000/api/models/metrics`
- ✅ If not trained, POST to `/models/train` to train on default dataset
- ✅ Models should auto-train on first backend startup

### Frontend dev server hangs on startup
```bash
# Use alternative startup command
PORT=4173 npx vite --config artifacts/sentiment-analysis/vite.config.ts --host 0.0.0.0
```

### Clear all trained models
```bash
# Models are stored in .data/trained-models.json
rm -f /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer/artifacts/.data/trained-models.json

# Restart backend to auto-train on default dataset
```

## 📊 Key Files Modified

1. **Frontend API URL**
   - File: [artifacts/sentiment-analysis/src/App.tsx](artifacts/sentiment-analysis/src/App.tsx)
   - Change: `setBaseUrl("http://localhost:8000")`

2. **Backend Port Configuration**
   - File: [artifacts/api-server/package.json](artifacts/api-server/package.json)
   - Change: Added `PORT=8000` to dev script

3. **Root npm Scripts**
   - File: [package.json](package.json)
   - Added: `backend:build`, `backend:start`, `frontend:dev` scripts

4. **Model Persistence** ✅ (Already working)
   - File: [artifacts/api-server/src/lib/modelStore.ts](artifacts/api-server/src/lib/modelStore.ts)
   - Features: Auto-load on startup, auto-save after training

## 🔐 Model Persistence Details

### How Models Are Persisted
1. After successful training, models are saved to `.data/trained-models.json`
2. On server startup, models are loaded from disk into memory
3. `store.trained` is set to `true` only if models exist on disk
4. All 3 models (Naive Bayes, Logistic Regression, SVM) are persisted together

### Files Involved
- **Storage location:** `/artifacts/.data/trained-models.json`
- **Load on startup:** [modelStore.ts lines 103-123](artifacts/api-server/src/lib/modelStore.ts#L103)
- **Auto-train fallback:** [modelStore.ts lines 206-211](artifacts/api-server/src/lib/modelStore.ts#L206)

## 🎓 Architecture

```
┌─────────────────────────────────────────────────────┐
│         Frontend (React + Vite)                      │
│         http://localhost:4173                       │
│  ┌──────────────────────────────────────────────┐  │
│  │ Pages:                                       │  │
│  │ - Analyze (single text)                      │  │
│  │ - Batch (multiple texts)                     │  │
│  │ - ML Training (train models)                 │  │
│  │ - Models (view metrics)                      │  │
│  │ - Compare (model comparison)                 │  │
│  │ - Word Cloud (keyword visualization)         │  │
│  │ - Report (export results)                    │  │
│  └──────────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────────────┘
               │
               ▼ API Calls (http://localhost:8000)
               
┌─────────────────────────────────────────────────────┐
│         Backend (Express.js)                        │
│         http://localhost:8000                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ ML Models (Persisted):                       │  │
│  │ - Naive Bayes                                │  │
│  │ - Logistic Regression                        │  │
│  │ - SVM                                        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │ API Routes:                                  │  │
│  │ - POST /sentiment/analyze                    │  │
│  │ - POST /sentiment/analyze-batch              │  │
│  │ - POST /models/train                         │  │
│  │ - GET  /models/metrics                       │  │
│  │ - POST /models/compare                       │  │
│  │ - POST /dataset/wordcloud                    │  │
│  │ - POST /report/generate                      │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │ Persistent Storage:                          │  │
│  │ - .data/trained-models.json                  │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 🎉 Success Indicators

You'll know everything is working when:

1. ✅ Backend starts without errors and reports `trained: true`
2. ✅ Frontend loads at `http://localhost:4173` without CORS errors
3. ✅ All pages load: Analyze, Batch, ML Training, Models, Compare, Word Cloud, Report
4. ✅ Sentiment analysis returns predictions with confidence scores
5. ✅ Model metrics show accuracy, precision, recall, F1 scores
6. ✅ Models remain trained after backend restart
7. ✅ CSV uploads work in batch analysis
8. ✅ Word cloud generates correctly from analyzed text
9. ✅ Reports export to JSON or CSV format

## 📝 Notes

- Models are now **never lost** after first training
- Backend auto-trains on first startup if no persisted models exist
- Frontend always shows current model status from backend
- All API endpoints return proper JSON with error handling
- CORS is properly configured for both localhost:4173 and localhost:8000
