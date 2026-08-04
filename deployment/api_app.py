"""
Spam Detection API - Fixed Version
Run: streamlit run deployment/api_app.py
"""
import streamlit as st
import joblib
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Disable sklearn warnings
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_or_train_model():
    """Load model if exists, otherwise train with consistent vectorizer"""
    try:
        # Try to load existing model
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        
        # Verify vectorizer works
        test_text = "test"
        test_vec = vectorizer.transform([test_text])
        print(f"✅ Model loaded: {test_vec.shape[1]} features")
        
        return model, vectorizer, "loaded"
        
    except Exception as e:
        st.warning(f"Model not found: {e}")
        st.info("⏳ Training new model with consistent vectorizer...")
        
        try:
            # Training data
            spam_emails = [
                "congratulations you won a prize claim your cash now",
                "free money click here to claim your reward",
                "urgent action required verify your account immediately",
                "your account has been compromised reset your password now",
                "limited time offer exclusive deal just for you",
                "earn 5000 dollars per day working from home",
                "investment opportunity guaranteed returns 100 percent",
                "credit card approved click here to activate",
                "loan approval guaranteed bad credit accepted",
                "mortgage refinance offer low interest rate",
                "insurance quote request urgent response needed",
                "medication at low cost no prescription required",
                "pharmacy discount free shipping worldwide",
                "weight loss miracle lose 20 pounds in 2 weeks",
                "anti aging breakthrough look 10 years younger",
                "free vacation to bahamas claim your trip now",
                "you have been selected for a special reward",
                "claim your free gift card worth 500 dollars",
                "exclusive offer for our valued customer",
                "don't miss this opportunity act fast",
                "guaranteed approval no credit check required",
                "lowest price guaranteed on all products",
                "free trial offer limited time only",
                "risk free opportunity try it now",
                "make money fast easy method revealed",
                "work from home job no experience needed",
                "passive income opportunity earn while you sleep",
                "financial freedom in 30 days program",
                "debt relief program eliminate your debt now",
                "cash bonus available for a limited time",
            ]
            
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
                "see you at the meeting tomorrow afternoon",
                "will send the documents shortly for your review",
                "please review the proposal and provide feedback",
                "looking forward to hearing from you soon",
                "lets touch base tomorrow to discuss next steps",
                "hope you are having a productive day",
                "thank you for your assistance with this project",
                "i will get back to you soon with the details",
                "we should catch up soon and discuss progress",
                "please confirm your availability for the meeting",
                "looking forward to working with you on this",
                "have a productive week ahead",
                "talk to you then about the project updates",
                "enjoy your weekend and take care",
                "take care of yourself and stay healthy",
            ]
            
            # Multiply data for better training
            all_spam = spam_emails * 3
            all_ham = ham_emails * 3
            
            df = pd.DataFrame({
                'text': all_spam + all_ham,
                'label': [1]*len(all_spam) + [0]*len(all_ham)
            })
            
            def clean_text(text):
                if not isinstance(text, str):
                    text = str(text)
                text = text.lower()
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            df['clean'] = df['text'].apply(clean_text)
            
            # Consistent vectorizer - FIXED NUMBER OF FEATURES
            vectorizer = TfidfVectorizer(
                max_features=2000,  # Consistent features
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85
            )
            
            X = vectorizer.fit_transform(df['clean'])
            y = df['label']
            
            st.info(f"📊 Training on {len(df)} emails with {X.shape[1]} features")
            
            # Train model
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X, y)
            
            # Calibrate
            calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
            calibrated_rf.fit(X, y)
            
            # Save
            os.makedirs('models', exist_ok=True)
            joblib.dump(calibrated_rf, 'models/classifier.pkl')
            joblib.dump(vectorizer, 'models/vectorizer.pkl')
            
            st.success(f"✅ Model trained on {len(df)} emails with {X.shape[1]} features!")
            return calibrated_rf, vectorizer, "trained"
            
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            return None, None, "error"

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_spam(email, model, vectorizer):
    if model is None or vectorizer is None:
        return {'error': 'Model not available'}
    
    try:
        clean = clean_text(email)
        vectorized = vectorizer.transform([clean])
        
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        confidence = float(max(probabilities))
        
        # Boost confidence for clear cases
        spam_words = ['won', 'free', 'prize', 'cash', 'money', 'urgent', 'verify', 'limited', 'offer', 'claim', 'winner']
        ham_words = ['meeting', 'project', 'thanks', 'report', 'review', 'please', 'team', 'how are you']
        
        text_lower = email.lower()
        spam_count = sum(1 for w in spam_words if w in text_lower)
        ham_count = sum(1 for w in ham_words if w in text_lower)
        
        # Boost confidence based on word counts
        if prediction == 1 and spam_count >= 2:
            confidence = min(confidence * 1.2, 0.99)
        elif prediction == 0 and ham_count >= 2:
            confidence = min(confidence * 1.2, 0.99)
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': confidence,
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0])
        }
        
    except Exception as e:
        return {'error': str(e)}

# Load or train model
with st.spinner("Loading/Training model..."):
    model, vectorizer, status = load_or_train_model()

st.set_page_config(page_title="Spam Detection API", page_icon="📧", layout="wide")

st.title("📧 Spam Detection API")
st.markdown("### Powered by Random Forest with Consistent Features")

with st.sidebar:
    st.subheader("📊 Status")
    if status == "loaded":
        st.success("✅ Model Loaded")
        st.info("🚀 Ready for predictions")
    elif status == "trained":
        st.success("✅ Model Trained Successfully!")
    else:
        st.error("❌ Model Not Available")

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
                    st.error(f"❌ {result['error']}")
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
