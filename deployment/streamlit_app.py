"""
Streamlit Dashboard for Spam Detection
Run: streamlit run deployment/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import joblib
import re
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_text(text):
    """Clean email text"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\$\d+\.?\d*', 'MONEY', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model():
    """Load model with caching"""
    try:
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

def predict_spam(text, model, vectorizer):
    """Predict if text is spam"""
    if model is None or vectorizer is None:
        return {'prediction': 'error', 'confidence': 0}
    
    clean = clean_text(text)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'confidence': float(max(probabilities))
    }

def main():
    """Main function to run the app"""
    
    st.set_page_config(
        page_title="Spam Detection Dashboard",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Spam Detection Dashboard")
    st.markdown("### Powered by Random Forest (97.34% Accuracy)")
    
    # Load model
    model, vectorizer = load_model()
    
    with st.sidebar:
        st.title("📊 Status")
        if model is not None:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Loaded")
            st.info("Training on first run may take a few minutes...")
    
    st.subheader("🔍 Check if an email is Spam or Ham")
    
    # Quick examples
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 Spam Example"):
            st.session_state.email_input = "Congratulations! You won $1,000,000! Click here to claim!"
    with col2:
        if st.button("📌 Ham Example"):
            st.session_state.email_input = "Hi, how are you doing today?"
    with col3:
        if st.button("📌 URGENT Example"):
            st.session_state.email_input = "URGENT: Your account has been compromised!"
    
    # Text input
    email_input = st.text_area(
        "Enter email content:",
        height=150,
        placeholder="Paste email text here...",
        key="email_input"
    )
    
    # Predict button
    if st.button("🔍 Predict", type="primary"):
        if email_input:
            with st.spinner("Analyzing..."):
                result = predict_spam(email_input, model, vectorizer)
                if result['prediction'] == 'spam':
                    st.error(f"⚠️ SPAM (Confidence: {result['confidence']:.2%})")
                elif result['prediction'] == 'ham':
                    st.success(f"✅ HAM (Confidence: {result['confidence']:.2%})")
                else:
                    st.warning("Model not ready. Please wait.")
        else:
            st.warning("Please enter some text")
    
    st.markdown("---")
    st.caption("🚀 Deployed on Streamlit Cloud")

if __name__ == "__main__":
    main()
