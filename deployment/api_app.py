"""
Spam Detection API - Fresh Start
"""
import streamlit as st
import joblib
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_or_train_model():
    """Load or train model"""
    try:
        # Try to load existing model
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer, "loaded"
    except:
        # Train new model
        st.info("⏳ Training model... This will take 1-2 minutes")
        
        # Training data
        spam = [
            "congratulations you won a prize claim your cash now",
            "free money click here to claim your reward",
            "urgent action required verify your account",
            "you have been selected for a special offer",
            "claim your cash prize now",
            "limited time offer exclusive deal",
            "winner winner chicken dinner",
            "click here to claim your reward",
            "your account has been compromised",
            "verify your identity immediately",
            "free gift card waiting for you",
            "you are the lucky winner",
            "exclusive deal just for you",
            "don't miss this opportunity",
            "cash bonus available now",
        ]
        
        ham = [
            "hi how are you doing today",
            "can we meet tomorrow to discuss the project",
            "thanks for your email i will review it",
            "please find attached the report",
            "good morning team here is the update",
            "let me know your thoughts on this",
            "looking forward to our meeting",
            "have a great day",
            "thanks for your help",
            "i appreciate your response",
            "can you send me the file",
            "lets schedule a call",
            "thank you for your time",
            "best regards",
            "have a wonderful weekend",
        ]
        
        # Multiply data
        df = pd.DataFrame({
            'text': spam*3 + ham*3,
            'label': [1]*len(spam*3) + [0]*len(ham*3)
        })
        
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text.strip()
        
        df['clean'] = df['text'].apply(clean_text)
        
        # Vectorizer with fixed features
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(df['clean'])
        y = df['label']
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Calibrate
        calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
        calibrated_rf.fit(X, y)
        
        # Save
        os.makedirs('models', exist_ok=True)
        joblib.dump(calibrated_rf, 'models/classifier.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        st.success(f"✅ Model trained on {len(df)} emails!")
        return calibrated_rf, vectorizer, "trained"

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def predict_spam(email, model, vectorizer):
    if model is None or vectorizer is None:
        return {'error': 'Model not available'}
    
    clean = clean_text(email)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    confidence = float(max(probabilities))
    
    # Boost confidence for common phrases
    ham_phrases = ['good morning', 'good afternoon', 'how are you', 'thanks', 'thank you']
    spam_phrases = ['won', 'free', 'prize', 'cash', 'money', 'urgent', 'verify', 'claim']
    
    text_lower = email.lower()
    
    if any(p in text_lower for p in ham_phrases):
        if prediction == 0:
            confidence = min(confidence * 1.3, 0.95)
    
    if any(p in text_lower for p in spam_phrases):
        if prediction == 1:
            confidence = min(confidence * 1.3, 0.95)
    
    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'confidence': confidence,
        'spam_probability': float(probabilities[1]),
        'ham_probability': float(probabilities[0])
    }

# Load model
model, vectorizer, status = load_or_train_model()

st.set_page_config(page_title="Spam Detection API", page_icon="📧", layout="wide")

st.title("📧 Spam Detection API")
st.markdown("### Powered by Random Forest")

with st.sidebar:
    st.subheader("📊 Status")
    if status == "loaded":
        st.success("✅ Model Loaded")
    elif status == "trained":
        st.success("✅ Model Trained")
    else:
        st.error("❌ Not Available")

st.subheader("🔍 Test the API")

email_input = st.text_area(
    "Enter email text:",
    height=100,
    placeholder="Paste email here..."
)

if st.button("🔍 Predict", type="primary"):
    if email_input:
        if model is not None:
            with st.spinner("Analyzing..."):
                result = predict_spam(email_input, model, vectorizer)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.json(result)
                    if result["prediction"] == "spam":
                        st.error(f"⚠️ SPAM (Confidence: {result['confidence']:.2%})")
                    else:
                        st.success(f"✅ HAM (Confidence: {result['confidence']:.2%})")
        else:
            st.warning("Model not ready")
    else:
        st.warning("Please enter some text")

st.markdown("---")
st.caption("🚀 Deployed on Streamlit Cloud")
