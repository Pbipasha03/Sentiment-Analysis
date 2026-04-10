from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from sentiment_core import SentimentService, ensure_texts_payload, normalize_dataframe


app = Flask(__name__)
CORS(app)

service = SentimentService()


@app.get("/health")
def health() -> tuple:
    return (
        jsonify(
            {
                "status": "ok",
                "models_trained": service.models_trained or service.load(),
                "message": "Sentiment API is running.",
            }
        ),
        200,
    )


@app.get("/metrics")
def metrics() -> tuple:
    try:
        return jsonify(service.metrics_payload()), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/train")
def train() -> tuple:
    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            file = request.files.get("file")
            if file is None or not file.filename:
                return jsonify({"error": "Upload a CSV file in form field 'file'."}), 400
            df = pd.read_csv(file)
        else:
            payload = request.get_json(silent=True) or {}
            dataset_path = payload.get("dataset_path")
            if dataset_path:
                df = pd.read_csv(Path(dataset_path))
            else:
                df = None

        artifacts = service.train(df)
        return (
            jsonify(
                {
                    "message": "Models trained successfully.",
                    "models_trained": True,
                    "best_model": artifacts.best_model_name,
                    "labels": artifacts.labels,
                    "metrics": artifacts.metrics,
                    "data_summary": artifacts.data_summary,
                }
            ),
            200,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict() -> tuple:
    try:
        payload = request.get_json(silent=True)
        texts = ensure_texts_payload(payload)
        model_name = payload.get("model")
        predictions = service.predict(texts, model_name=model_name)
        return (
            jsonify(
                {
                    "count": len(predictions),
                    "best_model": service.best_model_name,
                    "predictions": predictions,
                }
            ),
            200,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/predict-csv")
def predict_csv() -> tuple:
    try:
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error": "Upload a CSV file in form field 'file'."}), 400
        df = pd.read_csv(file)
        normalized_df, has_labels = normalize_dataframe(df)
        texts = normalized_df["text"].astype(str).tolist()
        predictions = service.predict(texts)
        response = {"count": len(predictions), "predictions": predictions}
        if has_labels:
            response["message"] = "CSV included labels; use /train to retrain models."
        return jsonify(response), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(404)
def not_found(_error) -> tuple:
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    service.ensure_trained()
    app.run(host="0.0.0.0", port=5000, debug=False)
