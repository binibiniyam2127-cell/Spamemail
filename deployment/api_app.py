import streamlit as st
import joblib
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer, "loaded"
    except:
        try:
            st.info("⏳ Training model... This will take 1-2 minutes")
            
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
                "winner winner you have been selected",
                "click here to claim your prize now",
                "you are the lucky winner of our lottery",
                "special promotion just for you today",
                "act fast limited stock available",
                "free iphone 15 click here to get yours",
                "win a brand new car enter now",
                "congratulations you are our winner",
                "claim your cash prize before it expires",
                "you have won a luxury vacation package",
                "free samples available order now",
                "exclusive membership discount for you",
                "urgent security alert verify your identity",
                "account locked due to suspicious activity",
                "your account is at risk please verify",
                "immediate action required to restore your account",
                "security breach detected secure your account",
                "unauthorized login attempt from new device",
                "your account has been suspended click to restore",
                "verify your email address to avoid closure",
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
                "keep up the good work on the project",
                "thanks for the update on the project status",
                "i understand your concern and will address it",
                "lets find a solution together as a team",
                "appreciate your patience and understanding",
                "will follow up next week with more details",
                "please let me know if you have any questions",
                "looking forward to your valuable feedback",
                "hope all is well with you and your team",
                "take care and stay safe during these times",
                "thanks for the detailed report it looks great",
                "i will review the document and get back to you",
                "can we schedule a meeting for this afternoon",
                "let me know when you are available to discuss",
                "thank you for your cooperation on this matter",
                "looking forward to a successful collaboration",
                "have a great rest of the week",
                "take care and talk to you soon",
                "appreciate your hard work on this project",
                "good luck with the presentation tomorrow",
            ]
            
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
            
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85
            )
            
            X = vectorizer.fit_transform(df['clean'])
            y = df['label']
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X, y)
            
            calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
            calibrated_rf.fit(X, y)
            
            os.makedirs('models', exist_ok=True)
            joblib.dump(calibrated_rf, 'models/classifier.pkl')
            joblib.dump(vectorizer, 'models/vectorizer.pkl')
            
            st.success(f"✅ Model trained on {len(df)} emails!")
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
        word_count = len(email.split())
        
        # ===== CONFIDENCE BOOST FOR SHORT TEXTS =====
        ham_keywords = ['hello', 'hi', 'hey', 'good', 'great', 'nice', 'thanks', 'thank', 'please', 'sorry', 
                       'how are you', 'how are you doing', 'good morning', 'good afternoon', 'good evening',
                       'best', 'regards', 'kind regards', 'sincerely', 'cheers', 'warmly']
        
        spam_keywords = ['won', 'free', 'prize', 'cash', 'money', 'urgent', 'verify', 'limited', 'offer', 
                        'claim', 'winner', 'click', 'win', 'gift', 'reward', 'bonus', 'deal', 'promotion',
                        'congratulations', 'guaranteed', 'exclusive', 'immediate', 'act now']
        
        text_lower = email.lower()
        
        # Count keywords
        ham_count = sum(1 for w in ham_keywords if w in text_lower)
        spam_count = sum(1 for w in spam_keywords if w in text_lower)
        
        # For very short texts (1-3 words) with clear ham keywords
        if word_count <= 3:
            if ham_count > 0 and spam_count == 0:
                return {
                    'prediction': 'ham',
                    'confidence': 0.85,
                    'spam_probability': 0.15,
                    'ham_probability': 0.85
                }
            elif spam_count > 0 and ham_count == 0:
                return {
                    'prediction': 'spam',
                    'confidence': 0.85,
                    'spam_probability': 0.85,
                    'ham_probability': 0.15
                }
            elif "good" in text_lower and len(text_lower) <= 10:
                # "good" is a positive ham word
                return {
                    'prediction': 'ham',
                    'confidence': 0.80,
                    'spam_probability': 0.20,
                    'ham_probability': 0.80
                }
        
        # Boost confidence for clear cases
        if prediction == 1 and spam_count >= 2:
            confidence = min(confidence * 1.3, 0.99)
        elif prediction == 0 and ham_count >= 2:
            confidence = min(confidence * 1.3, 0.99)
        elif word_count >= 10:
            # Longer texts get a confidence boost
            confidence = min(confidence * 1.2, 0.99)
        elif word_count <= 3 and confidence < 0.65:
            # Short ambiguous texts default to ham with moderate confidence
            confidence = 0.70
        
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
st.markdown("### Powered by Random Forest")

with st.sidebar:
    st.subheader("📊 Status")
    if status == "loaded":
        st.success("✅ Model Loaded")
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
