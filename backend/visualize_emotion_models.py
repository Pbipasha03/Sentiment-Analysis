"""
visualize_emotion_models.py
===========================
Generate visualizations for emotion model comparison:
- Per-model confusion matrices
- Accuracy comparison bar chart
- Per-class metrics comparison

Usage:
    python visualize_emotion_models.py

Output:
    - confusion_matrices_*.png
    - accuracy_comparison.html (interactive Plotly chart)
    - metrics_comparison.html (interactive per-class comparison)
"""

import json
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "metrics_summary.json"
VECTORIZER_PATH = ROOT_DIR / "emotion_vectorizer.pkl"
NB_MODEL_PATH = ROOT_DIR / "emotion_naive_bayes.pkl"
LR_MODEL_PATH = ROOT_DIR / "emotion_logistic_regression.pkl"
SVM_MODEL_PATH = ROOT_DIR / "emotion_svm.pkl"
EMOTIONS = ["happy", "sad", "angry", "fear", "neutral", "surprise"]


def plot_confusion_matrices():
    """Generate and save confusion matrices for each model."""
    print("\n📊 Generating confusion matrices...")
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for idx, model_metrics in enumerate(metrics["models"]):
        model_name = model_metrics["model"]
        cm = np.array(model_metrics["confusion_matrix"])
        
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=EMOTIONS,
            yticklabels=EMOTIONS,
            cbar_kws={"label": "Count"},
            ax=ax,
        )
        ax.set_title(f"{model_name}\n(Accuracy: {model_metrics['accuracy']:.2%})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Emotion", fontsize=10)
        ax.set_xlabel("Predicted Emotion", fontsize=10)
    
    plt.tight_layout()
    output_path = ROOT_DIR / "confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved confusion matrices to {output_path.name}")
    plt.close()


def plot_accuracy_comparison():
    """Generate interactive accuracy comparison bar chart."""
    print("📊 Generating accuracy comparison chart...")
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    df_accuracy = pd.DataFrame({
        "Model": [m["model"] for m in metrics["models"]],
        "Accuracy": [m["accuracy"] for m in metrics["models"]],
        "Macro F1": [m["f1_macro"] for m in metrics["models"]],
        "Weighted F1": [m["f1_weighted"] for m in metrics["models"]],
    })
    
    # Create stacked comparison
    fig = go.Figure()
    
    for metric in ["Accuracy", "Macro F1", "Weighted F1"]:
        fig.add_trace(go.Bar(
            x=df_accuracy["Model"],
            y=df_accuracy[metric],
            name=metric,
            text=[f"{v:.2%}" for v in df_accuracy[metric]],
            textposition="outside",
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode="group",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )
    
    output_path = ROOT_DIR / "accuracy_comparison.html"
    fig.write_html(str(output_path))
    print(f"  ✓ Saved to {output_path.name}")


def plot_per_class_metrics():
    """Generate per-class precision/recall/F1 comparison."""
    print("📊 Generating per-class metrics comparison...")
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    # Build dataframe for per-class metrics
    rows = []
    for model_metrics in metrics["models"]:
        model_name = model_metrics["model"]
        for emotion, class_data in model_metrics["class_report"].items():
            rows.append({
                "Model": model_name,
                "Emotion": emotion.capitalize(),
                "Precision": class_data["precision"],
                "Recall": class_data["recall"],
                "F1 Score": class_data["f1_score"],
            })
    
    df_classes = pd.DataFrame(rows)
    
    # Precision by emotion
    fig1 = px.bar(
        df_classes,
        x="Emotion",
        y="Precision",
        color="Model",
        barmode="group",
        title="Precision by Emotion Class",
        template="plotly_white",
        height=450,
    )
    fig1.write_html(str(ROOT_DIR / "metrics_precision.html"))
    
    # Recall by emotion
    fig2 = px.bar(
        df_classes,
        x="Emotion",
        y="Recall",
        color="Model",
        barmode="group",
        title="Recall by Emotion Class",
        template="plotly_white",
        height=450,
    )
    fig2.write_html(str(ROOT_DIR / "metrics_recall.html"))
    
    # F1 by emotion
    fig3 = px.bar(
        df_classes,
        x="Emotion",
        y="F1 Score",
        color="Model",
        barmode="group",
        title="F1-Score by Emotion Class",
        template="plotly_white",
        height=450,
    )
    fig3.write_html(str(ROOT_DIR / "metrics_f1.html"))
    
    print(f"  ✓ Saved precision, recall, and F1 visualizations")


def generate_comparison_table():
    """Print a comprehensive comparison table."""
    print("\n" + "="*90)
    print("EMOTION MODEL COMPARISON TABLE")
    print("="*90)
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    print(f"\nDataset: {metrics['dataset_size']} samples | Split: {metrics['train_test_split']}")
    print(f"Emotions: {', '.join(e.capitalize() for e in metrics['emotion_labels'])}")
    
    print("\n" + "-"*90)
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Train Time':<14} {'Inference':<10}")
    print("-"*90)
    
    for model_metrics in metrics["models"]:
        print(
            f"{model_metrics['model']:<20} "
            f"{model_metrics['accuracy']:<12.4f} "
            f"{model_metrics['f1_macro']:<12.4f} "
            f"{model_metrics['f1_weighted']:<12.4f} "
            f"{model_metrics['training_time_ms']:<13.2f}ms "
            f"{model_metrics['inference_time_ms']:<9.2f}ms"
        )
    
    print("-"*90)
    print(f"Best Model: {metrics['best_model']}\n")


if __name__ == "__main__":
    print("\n" + "="*90)
    print("EMOTION MODEL VISUALIZATION")
    print("="*90)
    
    generate_comparison_table()
    plot_confusion_matrices()
    plot_accuracy_comparison()
    plot_per_class_metrics()
    
    print("\n✓ All visualizations generated successfully!")
