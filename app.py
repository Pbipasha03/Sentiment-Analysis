import streamlit as st


def init_state():
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
    if "models" not in st.session_state:
        st.session_state.models = {}
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}
    if "labels" not in st.session_state:
        st.session_state.labels = []
    if "training_summary" not in st.session_state:
        st.session_state.training_summary = {}
    if "test_data" not in st.session_state:
        st.session_state.test_data = None


init_state()

import io
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_CANDIDATES = [
    ROOT_DIR / "microtext_sentiment_dataset.csv",
    ROOT_DIR / "backend" / "dataset.csv",
    ROOT_DIR / "backend" / "sample_dataset_expanded.csv",
    ROOT_DIR / "backend" / "sample_training_data.csv",
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label):
    label = str(label).strip().lower()
    label_map = {
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "1": "positive",
        "0": "negative",
        "2": "neutral",
    }
    return label_map.get(label, label)


def normalize_dataframe(df):
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    columns = {str(column).strip().lower(): column for column in df.columns}
    text_column = None
    label_column = None

    for candidate in ("text", "texts", "tweet", "content", "sentence", "review"):
        if candidate in columns:
            text_column = columns[candidate]
            break

    for candidate in ("sentiment", "label", "target", "class"):
        if candidate in columns:
            label_column = columns[candidate]
            break

    if text_column is None:
        text_column = df.columns[0]

    if label_column is None:
        raise ValueError("CSV must contain a sentiment, label, target, or class column.")

    normalized = pd.DataFrame(
        {
            "text": df[text_column].astype(str).fillna("").map(str.strip),
            "sentiment": df[label_column].astype(str).fillna("").map(normalize_label),
        }
    )
    normalized = normalized[(normalized["text"] != "") & (normalized["sentiment"] != "")]
    normalized = normalized.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    if normalized.empty:
        raise ValueError("Dataset has no valid text and sentiment rows.")

    return normalized


def load_default_dataset():
    for dataset_path in DEFAULT_DATASET_CANDIDATES:
        if dataset_path.exists():
            return normalize_dataframe(pd.read_csv(dataset_path))
    raise FileNotFoundError("No default CSV dataset found.")


def validate_training_data(texts, y):
    if len(texts) < 4:
        raise ValueError("Need at least 4 valid rows to train models.")

    label_counts = Counter(y)
    if len(label_counts) < 2:
        raise ValueError("Need at least 2 sentiment classes to train models.")

    too_small = [label for label, count in label_counts.items() if count < 2]
    if too_small:
        raise ValueError(
            "Each sentiment class needs at least 2 samples. Too few samples for: "
            + ", ".join(sorted(too_small))
        )


def train_session_models(df):
    normalized_df = normalize_dataframe(df)
    texts = [clean_text(text) for text in normalized_df["text"].tolist()]
    y = normalized_df["sentiment"].tolist()

    cleaned_pairs = [(text, label) for text, label in zip(texts, y) if text.strip()]
    if not cleaned_pairs:
        raise ValueError("No usable text remains after preprocessing.")

    texts = [text for text, _label in cleaned_pairs]
    y = [label for _text, label in cleaned_pairs]
    validate_training_data(texts, y)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=max(0.2, len(set(y)) / len(y)),
        stratify=y,
        random_state=42,
    )

    nb = MultinomialNB().fit(X_train, y_train)
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)

    st.session_state.models = {
        "Naive Bayes": nb,
        "Logistic Regression": lr,
        "SVM": svm,
    }

    st.session_state.vectorizer = vectorizer
    st.session_state.models_trained = True
    st.session_state.labels = sorted(set(y))
    st.session_state.test_data = {
        "X_test": X_test,
        "y_test": y_test,
    }
    st.session_state.training_summary = {
        "total_samples": len(y),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "classes": len(set(y)),
        "features": X.shape[1],
    }

    st.session_state.metrics = {}
    for model_name, model in st.session_state.models.items():
        predictions = model.predict(X_test)
        report = classification_report(
            y_test,
            predictions,
            labels=st.session_state.labels,
            output_dict=True,
            zero_division=0,
        )
        st.session_state.metrics[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=st.session_state.labels,
            ).tolist(),
            "class_report": report,
        }


def predict_one(input_text, selected_model):
    if not st.session_state.models_trained:
        st.error("Models not trained. Please train first.")
        st.stop()

    model = st.session_state.models[selected_model]
    vectorizer = st.session_state.vectorizer

    cleaned_input = clean_text(input_text)
    X_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(X_input)
    return prediction[0], cleaned_input


def render_training_page():
    st.header("Train Models")

    uploaded_file = st.file_uploader("Upload CSV with text and sentiment columns", type=["csv"])

    if uploaded_file is not None:
        dataset = normalize_dataframe(pd.read_csv(uploaded_file))
        st.success(f"Loaded {len(dataset)} rows from uploaded CSV.")
    else:
        dataset = load_default_dataset()
        st.info(f"Using default dataset with {len(dataset)} rows.")

    st.dataframe(dataset.head(20), use_container_width=True)

    if st.button("Train Models", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            train_session_models(dataset)
        st.success("Models trained successfully!")

    st.write("DEBUG:", st.session_state.models_trained)

    if st.session_state.models_trained:
        summary = st.session_state.training_summary
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Samples", summary["total_samples"])
        col2.metric("Train", summary["train_samples"])
        col3.metric("Test", summary["test_samples"])
        col4.metric("Classes", summary["classes"])
        col5.metric("Features", summary["features"])

        metrics_df = pd.DataFrame(
            [
                {
                    "Model": model_name,
                    "Accuracy": round(values["accuracy"] * 100, 2),
                    "Precision": round(values["precision"] * 100, 2),
                    "Recall": round(values["recall"] * 100, 2),
                    "F1": round(values["f1"] * 100, 2),
                }
                for model_name, values in st.session_state.metrics.items()
            ]
        )
        st.subheader("Model Metrics")
        st.dataframe(metrics_df, use_container_width=True)


def render_prediction_page():
    st.header("Predict Sentiment")
    st.write("DEBUG:", st.session_state.models_trained)

    if not st.session_state.models_trained:
        st.error("Models not trained. Please train first.")
        st.stop()

    selected_model = st.selectbox("Select model", list(st.session_state.models.keys()))
    input_text = st.text_area("Enter text", placeholder="Type a sentence to analyze...")

    if st.button("Predict", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter text to predict.")
            return

        prediction, cleaned_input = predict_one(input_text, selected_model)
        st.success(f"Prediction: {prediction}")
        st.write("Processed text:", cleaned_input)


def render_batch_page():
    st.header("Batch Prediction")
    st.write("DEBUG:", st.session_state.models_trained)

    if not st.session_state.models_trained:
        st.error("Models not trained. Please train first.")
        st.stop()

    selected_model = st.selectbox("Select model", list(st.session_state.models.keys()))
    uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])
    pasted_text = st.text_area("Or paste one text per line")

    texts = []
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_column = "text" if "text" in df.columns else df.columns[0]
        texts = df[text_column].astype(str).fillna("").tolist()
    elif pasted_text.strip():
        texts = [line.strip() for line in pasted_text.splitlines() if line.strip()]

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        if not texts:
            st.warning("Upload a CSV or paste at least one text.")
            return

        model = st.session_state.models[selected_model]
        vectorizer = st.session_state.vectorizer
        cleaned_texts = [clean_text(text) for text in texts]
        X_input = vectorizer.transform(cleaned_texts)
        predictions = model.predict(X_input)

        results_df = pd.DataFrame(
            {
                "text": texts,
                "processed_text": cleaned_texts,
                "prediction": predictions,
                "model": selected_model,
            }
        )
        st.success(f"Predicted {len(results_df)} rows.")
        st.dataframe(results_df, use_container_width=True)

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Predictions",
            data=csv_buffer.getvalue(),
            file_name="sentiment_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_metrics_page():
    st.header("Model Metrics")
    st.write("DEBUG:", st.session_state.models_trained)

    if not st.session_state.models_trained:
        st.error("Models not trained. Please train first.")
        st.stop()

    metrics_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy": values["accuracy"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["f1"],
            }
            for model_name, values in st.session_state.metrics.items()
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)

    labels = st.session_state.labels
    for model_name, values in st.session_state.metrics.items():
        st.subheader(f"{model_name} Confusion Matrix")
        cm_df = pd.DataFrame(values["confusion_matrix"], index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)


def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    init_state()

    st.title("Sentiment Analysis Streamlit App")
    st.write("DEBUG:", st.session_state.models_trained)

    page = st.sidebar.radio(
        "Page",
        ["Train Models", "Predict", "Batch Prediction", "Model Metrics"],
    )

    if page == "Train Models":
        render_training_page()
    elif page == "Predict":
        render_prediction_page()
    elif page == "Batch Prediction":
        render_batch_page()
    else:
        render_metrics_page()


if __name__ == "__main__":
    main()
