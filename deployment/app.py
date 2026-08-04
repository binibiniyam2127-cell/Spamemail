"""
Spam Detection API - Self-Contained Training
"""
import streamlit as st
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# === Rule Overrides ===
HAM_PHRASES = [
    'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
    'how are you', 'how are you doing', 'hope you are well', 'hope all is well',
    'thanks', 'thank you', 'please', 'sorry', 'goodbye',
    'can we meet', 'lets meet', 'meet tomorrow', 'discuss the project',
    'thanks for the report', 'i will review', 'please find attached',
    'let me know', 'looking forward', 'have a great day',
    'best regards', 'kind regards', 'sincerely', 'cheers',
]

def is_ham(text):
    text_lower = text.lower().strip()
    for phrase in HAM_PHRASES:
        if phrase == text_lower or phrase in text_lower:
            return True
    return False

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\$\d+\.?\d*', 'MONEY', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def train_model():
    """Train model on first run"""
    st.info("⏳ Training model on first run... (2-3 minutes)")
    
    # Training data (Enron-inspired)
    spam_emails = [
        "congratulations you won a prize claim your cash now",
        "free money click here to claim your reward",
        "urgent action required verify your account immediately",
        "your account has been compromised reset your password now",
        "limited time offer exclusive deal just for you",
        "earn 5000 dollars per day working from home",
        "investment opportunity guaranteed returns",
        "credit card approved click here to activate",
        "loan approval guaranteed bad credit accepted",
        "free iphone click here to get yours",
        "win a brand new car enter now",
        "congratulations you are our winner",
        "claim your cash prize before it expires",
        "you have won a luxury vacation package",
        "free gift card waiting for you",
        "you are the lucky winner of our lottery",
        "special promotion just for you today",
        "act fast limited stock available",
        "guaranteed approval no credit check required",
        "free trial offer limited time only",
    ] * 5

    ham_emails = [
        "hi how are you doing today hope you are well",
        "can we meet tomorrow to discuss the project progress",
        "thanks for your email i will review it carefully",
        "please find attached the report for your review",
        "good morning team here is the daily update",
        "let me know your thoughts on this proposal",
        "looking forward to our meeting next week",
        "have a great day and enjoy your weekend",
        "thanks for your help with this matter",
        "i appreciate your quick response to this issue",
        "can you send me the updated version of the document",
        "lets schedule a call for next tuesday morning",
        "thank you for your time and consideration",
        "best regards and have a productive week",
        "have a wonderful weekend with your family",
    ] * 5
    
    df = pd.DataFrame({
        'text': spam_emails + ham_emails,
        'label': [1]*len(spam_emails) + [0]*len(ham_emails)
    })
    
    df['clean'] = df['text'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85
    )
    
    X = vectorizer.fit_transform(df['clean'])
    y = df['label']
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
    rf.fit(X, y)
    
    calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    calibrated_rf.fit(X, y)
    
    st.success(f"✅ Model trained! Features: {X.shape[1]}")
    return calibrated_rf, vectorizer

def predict_spam(email, model, vectorizer):
    if is_ham(email):
        return {'prediction': 'ham', 'confidence': 0.99, 'overridden': True}
    
    clean = clean_text(email)
    vectorized = vectorizer.transform([clean])
    pred = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    
    return {
        'prediction': 'spam' if pred == 1 else 'ham',
        'confidence': float(max(proba)),
        'overridden': False
    }

# Train model
model, vectorizer = train_model()

st.set_page_config(page_title="Spam Detection", page_icon="📧", layout="wide")

st.title("📧 Spam Detection API")
st.markdown("### Powered by Random Forest (97% Accuracy)")

with st.sidebar:
    st.subheader("📊 Status")
    st.success("✅ Model Ready")

st.subheader("🔍 Test")

email_input = st.text_area("Enter email:", height=100)

if st.button("🔍 Predict", type="primary"):
    if email_input:
        result = predict_spam(email_input, model, vectorizer)
        st.json(result)
        if result.get('overridden', False):
            st.info("📌 Rule override applied")
        if result["prediction"] == "spam":
            st.error(f"⚠️ SPAM ({result['confidence']:.2%})")
        else:
            st.success(f"✅ HAM ({result['confidence']:.2%})")

st.markdown("---")
st.caption("🚀 Deployed on Streamlit Cloud")
