"""
Spam Detection Dashboard - Fixed Version
"""
import streamlit as st
import pandas as pd
import joblib
import re
import os
import numpy as np

st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="📧",
    layout="wide"
)

def clean_text(text):
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
    """Load model with error handling"""
    try:
        # Check if model exists
        if not os.path.exists('models/classifier.pkl'):
            return None, None, "Model not found"
        
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        
        # Check feature consistency
        try:
            # Test with a sample text
            test_text = "test email"
            test_clean = clean_text(test_text)
            test_vec = vectorizer.transform([test_clean])
            print(f"✅ Vectorizer works: {test_vec.shape[1]} features")
        except Exception as e:
            return None, None, f"Vectorizer error: {e}"
        
        return model, vectorizer, "loaded"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def predict_spam(text, model, vectorizer):
    if model is None or vectorizer is None:
        return {'prediction': 'error', 'confidence': 0, 'message': 'Model not loaded'}
    
    try:
        # Clean and vectorize
        clean = clean_text(text)
        vectorized = vectorizer.transform([clean])
        
        # Check if vectorized has features
        if vectorized.shape[1] == 0:
            return {'prediction': 'error', 'confidence': 0, 'message': 'No features extracted'}
        
        # Get prediction
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(max(probabilities)),
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0])
        }
    except Exception as e:
        return {'prediction': 'error', 'confidence': 0, 'message': str(e)}

# Load model
model, vectorizer, status = load_model()

# Title
st.title("📧 Spam Detection Dashboard")
st.markdown("### Powered by Random Forest + Enron Dataset")

# Sidebar
with st.sidebar:
    st.title("📊 Status")
    if status == "loaded":
        st.success("✅ Model Loaded")
        # Show feature count if available
        try:
            with open('models/feature_count.txt', 'r') as f:
                info = f.read()
                st.info(info)
        except:
            pass
    else:
        st.error(f"❌ {status}")

st.subheader("🔍 Check if an email is Spam or Ham")

# Quick examples
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📌 Spam"):
        st.session_state.email_input = "Congratulations! You won $1,000,000! Click here to claim!"
        st.rerun()
with col2:
    if st.button("📌 Ham"):
        st.session_state.email_input = "Hi, how are you doing today? Can we meet tomorrow?"
        st.rerun()
with col3:
    if st.button("📌 URGENT"):
        st.session_state.email_input = "URGENT: Your account has been compromised! Verify immediately!"
        st.rerun()

email_input = st.text_area(
    "Enter email content:",
    height=150,
    placeholder="Paste email text here...",
    key="email_input"
)

if st.button("🔍 Predict", type="primary"):
    if email_input:
        if model is not None and vectorizer is not None:
            with st.spinner("Analyzing..."):
                result = predict_spam(email_input, model, vectorizer)
                
                if result['prediction'] == 'spam':
                    st.error(f"⚠️ SPAM (Confidence: {result['confidence']:.2%})")
                elif result['prediction'] == 'ham':
                    st.success(f"✅ HAM (Confidence: {result['confidence']:.2%})")
                else:
                    st.warning(f"⚠️ {result.get('message', 'Unknown error')}")
        else:
            st.warning("Model not ready. Please wait for training to complete.")
    else:
        st.warning("Please enter some text")

st.markdown("---")
st.caption("🚀 Deployed on Streamlit Cloud")

# Show debug info in expander
with st.expander("🔧 Debug Info"):
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Vectorizer loaded: {vectorizer is not None}")
    if vectorizer is not None:
        try:
            test_vec = vectorizer.transform(["test"])
            st.write(f"Features: {test_vec.shape[1]}")
        except:
            st.write("Vectorizer test failed")
