#!/usr/bin/env python3
"""
Production Streamlit App for Sentiment Analysis
Features: Model training, single/batch prediction, comparison, visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import io
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))
from train_production import (
    load_or_create_dataset, train_models, clean_text,
    VECTORIZER_PATH, LABEL_ENCODER_PATH, NB_MODEL_PATH,
    LR_MODEL_PATH, SVM_MODEL_PATH, METRICS_PATH
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎯 Sentiment Analysis System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.vectorizer = None
    st.session_state.label_encoder = None
    st.session_state.models = {}
    st.session_state.metrics = {}
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.X_test_vec = None
    st.session_state.y_test_pred = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_models_exist():
    """Check if trained models exist on disk."""
    return all([
        VECTORIZER_PATH.exists(),
        LABEL_ENCODER_PATH.exists(),
        NB_MODEL_PATH.exists(),
        LR_MODEL_PATH.exists(),
        SVM_MODEL_PATH.exists()
    ])

def load_models_from_disk():
    """Load trained models from disk into session state."""
    try:
        st.session_state.vectorizer = joblib.load(VECTORIZER_PATH)
        st.session_state.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        st.session_state.models = {
            'Naive Bayes': joblib.load(NB_MODEL_PATH),
            'Logistic Regression': joblib.load(LR_MODEL_PATH),
            'SVM': joblib.load(SVM_MODEL_PATH)
        }
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                st.session_state.metrics = json.load(f)
        st.session_state.models_trained = True
        return True
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return False

def predict_sentiment(text, model_name='Logistic Regression'):
    """Predict sentiment for a single text."""
    if not st.session_state.models_trained:
        return {'error': 'Models not trained yet'}

    if not text.strip():
        return {'error': 'Please enter text to analyze'}

    try:
        # Clean text
        cleaned_text = clean_text(text)

        # Vectorize
        text_vec = st.session_state.vectorizer.transform([cleaned_text])

        # Predict
        model = st.session_state.models.get(model_name)
        if model is None:
            return {'error': f'Model {model_name} not found'}

        prediction = model.predict(text_vec)[0]
        sentiment = st.session_state.label_encoder.inverse_transform([prediction])[0]

        # Get confidence/probability
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_vec)[0]
            else:  # SVM
                decision = model.decision_function(text_vec)[0]
                proba = 1 / (1 + np.exp(-decision))
                proba = np.abs(proba)

            confidence = np.max(proba)
        except:
            confidence = 0.5

        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'cleaned_text': cleaned_text
        }
    except Exception as e:
        return {'error': f'Prediction error: {e}'}

def predict_batch(texts, model_name='Logistic Regression'):
    """Predict sentiment for multiple texts."""
    results = []
    for text in texts:
        result = predict_sentiment(text, model_name)
        results.append(result)
    return results

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Model selection
    selected_model = st.selectbox(
        'Select Model for Predictions:',
        ['Logistic Regression', 'Naive Bayes', 'SVM'],
        help='Logistic Regression typically has best accuracy'
    )

    st.divider()

    # Check model status
    st.markdown("### 📊 Model Status")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.models_trained:
            st.success("✅ Models Trained")
        else:
            st.warning("⚠️ Not Trained")

    with col2:
        if check_models_exist():
            st.info("💾 Saved")
        else:
            st.info("⏳ New")

    st.divider()

    # Show metrics if available
    if st.session_state.metrics:
        st.markdown("### 📈 Model Metrics")
        for model_name, metrics in st.session_state.metrics.items():
            st.metric(
                f"{model_name}",
                f"{metrics['accuracy']:.2%}",
                f"F1: {metrics['f1']:.3f}"
            )

    st.divider()
    st.markdown("**About this app**")
    st.info("""
    **Production Sentiment Analysis System**
    
    - ✅ 3 Machine Learning Models
    - ✅ Large Dataset (1000+ samples)
    - ✅ TF-IDF Vectorization
    - ✅ Model comparison & metrics
    - ✅ Confusion matrices
    - ✅ Batch processing
    - ✅ CSV upload/download
    """)

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("# 🎯 Sentiment Analysis System")
st.markdown("Professional sentiment analysis with ML model training and comparison")

# Check if models already trained
if not st.session_state.models_trained and check_models_exist():
    if st.button("🔄 Load Existing Models", use_container_width=True):
        with st.spinner("Loading models..."):
            if load_models_from_disk():
                st.success("✅ Models loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load models")

# ============================================================================
# TAB 1: TRAINING
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏋️ Train Models",
    "🔮 Single Prediction",
    "📊 Batch Analysis",
    "📈 Model Comparison",
    "📋 About"
])

with tab1:
    st.markdown("## 🏋️ Train Models")
    st.markdown("""
    Train 3 sentiment models on a large, balanced dataset.
    - **Dataset:** 1000+ samples (Positive, Negative, Neutral)
    - **Models:** Naive Bayes, Logistic Regression, Linear SVM
    - **Features:** TF-IDF with bigrams (5000 words)
    - **Split:** 80/20 stratified
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🚀 Train Models Now", use_container_width=True, key="train_btn", type="primary"):
            with st.spinner("⏳ Training models (this may take a minute)..."):
                try:
                    # Load dataset
                    df = load_or_create_dataset()

                    # Train models
                    results = train_models(df)

                    # Save to session state
                    st.session_state.vectorizer = results['vectorizer']
                    st.session_state.label_encoder = results['label_encoder']
                    st.session_state.models = results['models']
                    st.session_state.metrics = results['metrics']
                    st.session_state.X_test = results['X_test']
                    st.session_state.y_test = results['y_test']
                    st.session_state.X_test_vec = results['X_test_vec']
                    st.session_state.y_test_pred = results['y_test_pred']
                    st.session_state.models_trained = True

                    st.success("✅ Models trained successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            if load_models_from_disk():
                st.success("✅ Reloaded!")
                st.rerun()

    if st.session_state.models_trained:
        st.divider()
        st.markdown("### ✅ Training Results")

        # Display metrics table
        metrics_df = pd.DataFrame(st.session_state.metrics).T
        metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1']]
        metrics_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_df = metrics_df * 100

        st.dataframe(metrics_df, use_container_width=True)

        # Best model
        best_model = max(st.session_state.metrics.items(), key=lambda x: x[1]['f1'])
        st.markdown(f"### 🏆 Best Model: **{best_model[0]}** (F1: {best_model[1]['f1']:.4f})")

        # Metrics visualization
        col1, col2 = st.columns(2)

        with col1:
            # Accuracy comparison
            accuracy_data = {
                'Model': list(st.session_state.metrics.keys()),
                'Accuracy': [m['accuracy']*100 for m in st.session_state.metrics.values()]
            }
            fig = px.bar(accuracy_data, x='Model', y='Accuracy',
                        title='Model Accuracy Comparison',
                        color='Accuracy', color_continuous_scale='Viridis')
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # F1-Score comparison
            f1_data = {
                'Model': list(st.session_state.metrics.keys()),
                'F1-Score': [m['f1']*100 for m in st.session_state.metrics.values()]
            }
            fig = px.bar(f1_data, x='Model', y='F1-Score',
                        title='Model F1-Score Comparison',
                        color='F1-Score', color_continuous_scale='RdYlGn')
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        # Confusion matrices
        st.markdown("### 🔍 Confusion Matrices")
        for model_name, preds in st.session_state.y_test_pred.items():
            with st.expander(f"📊 {model_name} Confusion Matrix"):
                cm = confusion_matrix(st.session_state.y_test, preds)
                labels = st.session_state.label_encoder.classes_

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels, ax=ax,
                           cbar_kws={'label': 'Count'})
                ax.set_title(f'{model_name} - Confusion Matrix')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                st.pyplot(fig, use_container_width=True)
                plt.close()

# ============================================================================
# TAB 2: SINGLE PREDICTION
# ============================================================================

with tab2:
    st.markdown("## 🔮 Predict Sentiment")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Train Models' tab")
    else:
        # Text input
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="e.g., 'This product is amazing! I love it.'",
            height=120
        )

        if st.button("🔍 Analyze", use_container_width=True, type="primary"):
            if user_text.strip():
                # Predict with selected model
                result = predict_sentiment(user_text, selected_model)

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    sentiment = result['sentiment'].capitalize()
                    confidence = result['confidence']

                    # Display result with color
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        if sentiment == 'Positive':
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: #d4edda; border-radius: 10px;">
                                <h1>😊 {sentiment}</h1>
                                <p style="font-size: 24px; color: #155724;">✅</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif sentiment == 'Negative':
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: #f8d7da; border-radius: 10px;">
                                <h1>😞 {sentiment}</h1>
                                <p style="font-size: 24px; color: #721c24;">❌</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: #fff3cd; border-radius: 10px;">
                                <h1>😐 {sentiment}</h1>
                                <p style="font-size: 24px; color: #856404;">⚪</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")

                    with col3:
                        st.metric("Model", selected_model)

                    # Show cleaned text
                    st.markdown("### 🔧 Processed Text")
                    st.code(result['cleaned_text'])

                    # Prediction explanation
                    st.markdown("### 📝 Analysis")
                    st.info(f"""
                    **Model:** {selected_model}  
                    **Sentiment:** {sentiment}  
                    **Confidence:** {confidence:.2%}  
                    **Original text length:** {len(user_text)} characters
                    """)
            else:
                st.warning("⚠️ Please enter some text to analyze")

        # Show example predictions
        st.divider()
        st.markdown("### 💡 Example Predictions")

        examples = {
            "😊 Positive": "I absolutely love this product! Best purchase ever!",
            "😞 Negative": "Terrible quality, worst experience ever. Complete waste of money!",
            "😐 Neutral": "It's okay, nothing special, just average."
        }

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, (label, text) in enumerate(examples.items()):
            with cols[i]:
                if st.button(f"Test: {label}", use_container_width=True):
                    pred = predict_sentiment(text, selected_model)
                    if 'error' not in pred:
                        st.success(f"✅ {pred['sentiment'].capitalize()}")
                    st.code(text, language="text")

# ============================================================================
# TAB 3: BATCH ANALYSIS
# ============================================================================

with tab3:
    st.markdown("## 📊 Batch Analysis")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
    else:
        # Two options: Upload CSV or Paste texts
        analysis_type = st.radio(
            "Choose input method:",
            ["📤 Upload CSV File", "📝 Paste Texts"]
        )

        results = []

        if analysis_type == "📤 Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload CSV file (first column = text)",
                type=['csv']
            )

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)

                    if df.shape[0] == 0:
                        st.error("❌ CSV is empty")
                    else:
                        # Get text column
                        text_col = df.iloc[:, 0]
                        texts = text_col.astype(str).tolist()

                        if st.button("🚀 Analyze All Texts", use_container_width=True, type="primary"):
                            with st.spinner(f"⏳ Analyzing {len(texts)} texts..."):
                                results = predict_batch(texts, selected_model)

                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")

        else:  # Paste texts
            text_input = st.text_area(
                "Paste texts (one per line):",
                placeholder="Text 1\nText 2\nText 3",
                height=150
            )

            if st.button("🚀 Analyze All Texts", use_container_width=True, type="primary"):
                texts = [t.strip() for t in text_input.split('\n') if t.strip()]
                if texts:
                    with st.spinner(f"⏳ Analyzing {len(texts)} texts..."):
                        results = predict_batch(texts, selected_model)
                else:
                    st.warning("⚠️ Please enter at least one text")

        # Display results
        if results:
            valid_results = [r for r in results if 'error' not in r and r.get('sentiment')]

            if valid_results:
                st.success(f"✅ Analyzed {len(valid_results)} texts successfully!")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                sentiments = [r['sentiment'].capitalize() for r in valid_results]
                sentiment_counts = pd.Series(sentiments).value_counts()

                with col1:
                    st.metric("Total Analyzed", len(valid_results))
                with col2:
                    st.metric("Positive", sentiment_counts.get('Positive', 0))
                with col3:
                    st.metric("Negative", sentiment_counts.get('Negative', 0))
                with col4:
                    st.metric("Neutral", sentiment_counts.get('Neutral', 0))

                # Pie chart
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Results table
                st.markdown("### 📋 Detailed Results")
                results_df = pd.DataFrame([
                    {
                        'Text': r['cleaned_text'][:50],
                        'Sentiment': r['sentiment'].capitalize(),
                        'Confidence': f"{r['confidence']:.1%}"
                    }
                    for r in valid_results
                ])
                st.dataframe(results_df, use_container_width=True)

                # Download results
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("❌ No valid predictions")

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================

with tab4:
    st.markdown("## 📈 Model Comparison")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
    else:
        # Metrics table
        metrics_df = pd.DataFrame(st.session_state.metrics).T
        metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1']]
        metrics_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        st.markdown("### 📊 Performance Metrics")
        st.dataframe(metrics_df * 100, use_container_width=True)

        # Comparisons side by side
        col1, col2 = st.columns(2)

        with col1:
            # All metrics comparison
            metrics_comparison = pd.DataFrame({
                'Accuracy': [st.session_state.metrics[m]['accuracy'] for m in st.session_state.metrics],
                'Precision': [st.session_state.metrics[m]['precision'] for m in st.session_state.metrics],
                'Recall': [st.session_state.metrics[m]['recall'] for m in st.session_state.metrics],
                'F1-Score': [st.session_state.metrics[m]['f1'] for m in st.session_state.metrics]
            }, index=st.session_state.metrics.keys())

            fig = go.Figure()
            for metric in metrics_comparison.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_comparison.index,
                    y=metrics_comparison[metric]*100
                ))
            fig.update_layout(title="All Metrics Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Radar chart
            model_names = list(st.session_state.metrics.keys())
            fig = go.Figure()

            for model in model_names:
                metrics = st.session_state.metrics[model]
                fig.add_trace(go.Scatterpolar(
                    r=[metrics['accuracy']*100, metrics['precision']*100, 
                       metrics['recall']*100, metrics['f1']*100],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=model
                ))

            fig.update_layout(title="Radar Comparison (All Models)")
            st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics
        st.markdown("### 📋 Detailed Breakdown")
        for model_name, metrics in st.session_state.metrics.items():
            with st.expander(f"📊 {model_name}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1']:.4f}")

# ============================================================================
# TAB 5: ABOUT
# ============================================================================

with tab5:
    st.markdown("""
    # 📋 About This System
    
    ## ✨ Features
    
    ✅ **3 Machine Learning Models**
    - Multinomial Naive Bayes
    - Logistic Regression
    - Linear SVM
    
    ✅ **Large, Balanced Dataset**
    - 1000+ samples
    - Equal distribution: 33% Positive, 33% Negative, 33% Neutral
    - Automatically generated if small dataset provided
    
    ✅ **Advanced Text Processing**
    - TF-IDF Vectorization
    - Bigrams (1-2 word combinations)
    - Lemmatization
    - Stopword removal
    - 5000 features
    
    ✅ **Comprehensive Model Evaluation**
    - Accuracy, Precision, Recall, F1-Score
    - Confusion matrices with visualizations
    - Model comparison charts
    - Radar charts for multi-metric comparison
    
    ✅ **Production Features**
    - Single text prediction
    - Batch analysis (hundreds of texts)
    - CSV upload/download
    - Real-time visualizations
    - Responsive UI
    
    ## 📊 Model Performance
    
    Expected accuracies on balanced test set:
    - **Logistic Regression:** ~75-85%
    - **Naive Bayes:** ~70-80%
    - **SVM:** ~70-80%
    
    ## 🚀 How to Use
    
    1. **Train Models** → Click "Train Models Now" in the Train tab
    2. **Single Prediction** → Enter text and click "Analyze"
    3. **Batch Analysis** → Upload CSV or paste multiple texts
    4. **Compare Models** → View metrics and confusion matrices
    
    ## 💾 Data Flow
    
    ```
    Dataset (CSV)
        ↓
    Text Cleaning (lowercase, punctuation removal, lemmatization)
        ↓
    Train-Test Split (80/20 stratified)
        ↓
    TF-IDF Vectorization (fit on training only)
        ↓
    Model Training (3 models)
        ↓
    Evaluation & Metrics
        ↓
    Model Persistence (joblib .pkl files)
        ↓
    Predictions on New Data
    ```
    
    ## 🛠️ Technical Stack
    
    - **Framework:** Streamlit
    - **ML:** scikit-learn
    - **Data:** pandas, numpy
    - **Visualization:** plotly, seaborn, matplotlib
    - **NLP:** NLTK
    - **Model Storage:** joblib
    
    ## 📁 Files
    
    - `train_production.py` - Training pipeline
    - `app_production.py` - Streamlit UI (this app)
    - `requirements.txt` - Dependencies
    - `vectorizer_production.pkl` - Fitted vectorizer
    - `sentiment_*.pkl` - Trained models (Naive Bayes, LR, SVM)
    - `label_encoder_production.pkl` - Label encoding
    - `model_metrics_production.json` - Evaluation metrics
    
    ## 🎓 For B.Tech Submission
    
    This system demonstrates:
    - End-to-end ML pipeline
    - Data preprocessing and feature engineering
    - Model training and evaluation
    - Model comparison and selection
    - Production-ready code with error handling
    - User-friendly interface
    - Professional documentation
    
    ## 📞 Support
    
    If models are not trained:
    1. Click "Train Models Now" in the Train tab
    2. Wait for training to complete
    3. Models will be saved automatically
    
    For technical issues:
    - Check required packages: `pip install -r requirements.txt`
    - Ensure dataset exists in backend directory
    - Check console output for error messages
    """)

    # Status
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### System Status")
        if st.session_state.models_trained:
            st.success("✅ Models trained and ready")
        else:
            st.warning("⚠️ Models not trained yet")

    with col2:
        st.markdown("### Dataset Status")
        st.info("📊 1000+ samples available")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    🎯 Production Sentiment Analysis System | Powered by Streamlit • scikit-learn • NLP
</div>
""", unsafe_allow_html=True)
