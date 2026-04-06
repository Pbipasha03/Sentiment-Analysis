# How to Connect Your Frontend to This Backend API

## Your Project Folder Structure

```
your-project/
├── frontend/          ← Your existing React / HTML / JS files (do NOT touch)
│   ├── index.html
│   ├── App.js / App.jsx
│   └── ...
└── backend/           ← New folder (everything is here)
    ├── train_model.py
    ├── app.py
    ├── requirements.txt
    ├── dataset.csv
    ├── model.pkl       ← created after training
    └── vectorizer.pkl  ← created after training
```

---

## Step 1 — Install Python Libraries

Open a terminal, navigate to the backend folder, and run:

```bash
cd backend
pip install -r requirements.txt
```

---

## Step 2 — Train the Models

Make sure your dataset.csv is inside the backend folder.
The CSV must have two columns: **text** and **label** (positive / negative / neutral).

```bash
cd backend
python train_model.py
```

You will see accuracy for both models and a classification report.
Two files will be created: **model.pkl** and **vectorizer.pkl**

---

## Step 3 — Start the API Server

```bash
cd backend
python app.py
```

The server starts at:  http://127.0.0.1:5000

Your frontend should be served from `http://localhost:5173`.

Test it quickly:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Expected response:
```json
{ "sentiment": "positive", "original_text": "I love this product!", "clean_text": "love product" }
```

---

## Step 4 — Connect Your Existing Frontend

Paste this snippet wherever you need sentiment analysis in your existing JS/JSX code.
You do NOT need to rebuild or change anything else.

### Plain HTML / Vanilla JavaScript

```html
<!-- Add this inside any existing HTML file -->
<input type="text" id="textInput" placeholder="Enter text to analyze..." />
<button onclick="analyzeSentiment()">Analyze</button>
<p id="result"></p>

<script>
  async function analyzeSentiment() {
    const text = document.getElementById("textInput").value;

    if (!text.trim()) {
      alert("Please enter some text.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });

      const data = await response.json();
      document.getElementById("result").textContent = "Sentiment: " + data.sentiment;

    } catch (error) {
      console.error("API error:", error);
      alert("Could not connect to the backend. Make sure app.py is running.");
    }
  }
</script>
```

---

### React / JSX

Add this function inside any existing component:

```jsx
import { useState } from "react";

function SentimentAnalyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });

      const data = await response.json();
      setResult(data.sentiment);         // "positive" / "negative" / "neutral"

    } catch (error) {
      console.error("API error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to analyze..."
      />
      <button onClick={analyzeSentiment} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>
      {result && <p>Sentiment: <strong>{result}</strong></p>}
    </div>
  );
}

export default SentimentAnalyzer;
```

---

## API Reference

| Field     | Value                     |
|-----------|---------------------------|
| URL       | http://localhost:5000/predict |
| Method    | POST                      |
| Input     | `{ "text": "your sentence" }` |
| Output    | `{ "sentiment": "positive" }` |

### Possible sentiment values
- `positive`
- `negative`
- `neutral`

---

## Common Issues

| Problem | Fix |
|---|---|
| `CORS error` in browser | Make sure `flask-cors` is installed and `CORS(app)` is in app.py |
| `Failed to fetch` | Make sure `python app.py` is running |
| `model.pkl not found` | Run `train_model.py` first before starting `app.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
