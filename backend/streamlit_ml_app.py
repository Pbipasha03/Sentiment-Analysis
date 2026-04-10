from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

from sentiment_core import PREPROCESSOR, SentimentService, normalize_dataframe


API_BASE_URL = "http://localhost:5000"


def init_state() -> None:
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
    if "models" not in st.session_state:
        st.session_state.models = {}
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "service" not in st.session_state:
        st.session_state.service = SentimentService()
    if "metrics_payload" not in st.session_state:
        st.session_state.metrics_payload = None
    if "active_dataframe" not in st.session_state:
        st.session_state.active_dataframe = None
    if "labels" not in st.session_state:
        st.session_state.labels = []


init_state()


def sync_models_from_service() -> None:
    service = st.session_state.service
    if not service.models or service.vectorizer is None:
        return
    st.session_state.models = {
        "Naive Bayes": service.models["MultinomialNB"],
        "Logistic Regression": service.models["LogisticRegression"],
        "SVM": service.models["LinearSVC"],
    }
    st.session_state.vectorizer = service.vectorizer
    st.session_state.labels = service.labels
    st.session_state.models_trained = True
    st.session_state.metrics_payload = service.metrics_payload()


def ensure_models_ready() -> None:
    if st.session_state.models_trained and st.session_state.models and st.session_state.vectorizer is not None:
        return
    try:
        if st.session_state.service.load():
            sync_models_from_service()
    except Exception:
        st.session_state.models_trained = False
        st.session_state.models = {}
        st.session_state.vectorizer = None
        st.session_state.metrics_payload = None
        st.session_state.labels = []


def train_models(df: pd.DataFrame | None = None) -> None:
    with st.spinner("Training models..."):
        st.session_state.service.train(df)
    sync_models_from_service()


def parse_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(uploaded_file)
    normalized_df, has_labels = normalize_dataframe(df)
    return normalized_df, has_labels


def render_metrics_table(metrics_payload: dict) -> None:
    best_model = metrics_payload["best_model"]
    rows = []
    for model_name, values in metrics_payload["metrics"].items():
        display_name = {
            "MultinomialNB": "Naive Bayes",
            "LogisticRegression": "Logistic Regression",
            "LinearSVC": "SVM",
        }.get(model_name, model_name)
        rows.append(
            {
                "Model": display_name,
                "Accuracy": round(values["accuracy"] * 100, 2),
                "Precision": round(values["precision"] * 100, 2),
                "Recall": round(values["recall"] * 100, 2),
                "F1": round(values["f1"] * 100, 2),
                "Best": "Yes"
                if display_name
                == {
                    "MultinomialNB": "Naive Bayes",
                    "LogisticRegression": "Logistic Regression",
                    "LinearSVC": "SVM",
                }.get(best_model, best_model)
                else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_confusion_matrices(labels: list[str], metrics_payload: dict) -> None:
    for model_name, values in metrics_payload["metrics"].items():
        display_name = {
            "MultinomialNB": "Naive Bayes",
            "LogisticRegression": "Logistic Regression",
            "LinearSVC": "SVM",
        }.get(model_name, model_name)
        st.subheader(f"{display_name} Confusion Matrix")
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            pd.DataFrame(values["confusion_matrix"], index=labels, columns=labels),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        st.pyplot(figure)
        plt.close(figure)


def call_backend_prediction(texts: list[str], model_name: str) -> dict | None:
    backend_model_map = {
        "Naive Bayes": "MultinomialNB",
        "Logistic Regression": "LogisticRegression",
        "SVM": "LinearSVC",
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"texts": texts, "model": backend_model_map.get(model_name, model_name)},
            timeout=10,
        )
        if response.ok:
            return response.json()
    except requests.RequestException:
        return None
    return None


def predict_text(input_text: str, selected_model: str) -> dict:
    if not st.session_state.models_trained:
        st.error("Models not trained. Please train first.")
        st.stop()

    model = st.session_state.models.get(selected_model)
    vectorizer = st.session_state.vectorizer

    if model is None or vectorizer is None:
        st.error("Models not trained. Please train first.")
        st.stop()

    cleaned_text = PREPROCESSOR.clean(input_text)
    X_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(X_input)

    probabilities = None
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(X_input)[0]
        probabilities = {
            str(label): float(raw_probabilities[index])
            for index, label in enumerate(model.classes_)
        }
    else:
        backend_response = call_backend_prediction([input_text], selected_model)
        if backend_response:
            probabilities = backend_response["predictions"][0]["probabilities"]
        else:
            probabilities = {
                label: 1.0 if label == str(prediction[0]) else 0.0
                for label in st.session_state.labels
            }

    return {
        "prediction": str(prediction[0]),
        "confidence": float(probabilities.get(str(prediction[0]), 0.0)),
        "probabilities": probabilities,
    }


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis", page_icon="🤖", layout="wide")
    init_state()
    ensure_models_ready()

    st.title("Sentiment Analysis Platform")
    st.write("DEBUG:", st.session_state.models_trained)

    with st.sidebar:
        st.header("API Format")
        st.code(
            "fetch('http://localhost:5000/predict', {\n"
            "  method: 'POST',\n"
            "  headers: {'Content-Type': 'application/json'},\n"
            "  body: JSON.stringify({ texts: ['sample text'] })\n"
            "})",
            language="javascript",
        )

    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    left_column, right_column = st.columns([2, 1])

    with left_column:
        if uploaded_file is not None:
            try:
                normalized_df, has_labels = parse_uploaded_csv(uploaded_file)
                st.session_state.active_dataframe = normalized_df
                st.success(f"Loaded {len(normalized_df)} rows.")
                st.dataframe(normalized_df.head(10), use_container_width=True)
                if not has_labels:
                    st.info("Single-column CSV detected. This file can be used for batch prediction.")
            except Exception as exc:
                st.session_state.active_dataframe = None
                st.error(str(exc))
        else:
            st.info("No file uploaded. Built-in dataset will be used for training.")

    with right_column:
        if st.button("Train Models", use_container_width=True, type="primary"):
            try:
                training_df = None
                if (
                    st.session_state.active_dataframe is not None
                    and "sentiment" in st.session_state.active_dataframe.columns
                ):
                    training_df = st.session_state.active_dataframe
                train_models(training_df)
                st.success("Models trained successfully!")
            except Exception as exc:
                st.error(str(exc))

        st.write("DEBUG:", st.session_state.models_trained)

        if st.session_state.models_trained and st.session_state.metrics_payload:
            summary = st.session_state.metrics_payload["data_summary"]
            st.metric("Samples", summary["total_samples"])
            st.metric("Classes", summary["num_classes"])
            st.metric("Features", summary["num_features"])

    if st.session_state.models_trained and st.session_state.metrics_payload:
        st.header("Model Comparison")
        render_metrics_table(st.session_state.metrics_payload)
        render_confusion_matrices(st.session_state.labels, st.session_state.metrics_payload)

    st.header("Single Prediction")
    input_text = st.text_area("Enter text", placeholder="Type text to analyze...")
    selected_model = st.selectbox(
        "Select model",
        ["Naive Bayes", "Logistic Regression", "SVM"],
    )

    if st.button("Predict", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            result = predict_text(input_text, selected_model)
            st.success(f"Prediction: {result['prediction']}")
            first_metric, second_metric = st.columns(2)
            first_metric.metric("Prediction", result["prediction"])
            second_metric.metric("Confidence", f"{result['confidence']:.2%}")
            probability_df = pd.DataFrame(
                {
                    "Sentiment": list(result["probabilities"].keys()),
                    "Probability": [
                        round(value * 100, 2) for value in result["probabilities"].values()
                    ],
                }
            )
            st.dataframe(probability_df, use_container_width=True)

    st.header("Batch Prediction")
    if uploaded_file is not None and st.session_state.active_dataframe is not None:
        if "sentiment" not in st.session_state.active_dataframe.columns:
            if st.button("Run Batch Prediction", use_container_width=True):
                if not st.session_state.models_trained:
                    st.error("Models not trained. Please train first.")
                    st.stop()
                texts = st.session_state.active_dataframe["text"].astype(str).tolist()
                predictions = st.session_state.service.predict(texts)
                prediction_df = pd.DataFrame(predictions)
                st.dataframe(prediction_df, use_container_width=True)
                csv_buffer = io.StringIO()
                prediction_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download Predictions",
                    data=csv_buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Uploaded CSV includes sentiment labels and is ready for training.")


if __name__ == "__main__":
    main()
