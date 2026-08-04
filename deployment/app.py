"""
Spam Detection API - Self-Contained (No External Models)
"""
import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# === Rule Overrides ===
HAM_WORDS = ['hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening',
             'thanks', 'thank', 'please', 'sorry', 'meet', 'meeting',
             'project', 'report', 'review', 'team', 'weekend', 'today',
             'tomorrow', 'yesterday', 'monday', 'tuesday', 'wednesday',
             'thursday', 'friday', 'saturday', 'sunday']

def is_ham_override(text):
    text_lower = text.lower().strip()
    if text_lower in ['hello', 'hi', 'hey', 'thanks', 'thank you']:
        return True
    for word in HAM_WORDS:
        if word in text_lower:
            return True
    return False

@st.cache_resource
def get_model():
    """Train model on first run - no external files needed"""
    st.info("⏳ Loading spam detection model...")
    
    # Training data
    spam = [
        "congratulations you won a prize claim your cash now",
        "free money click here to claim your reward",
        "urgent action required verify your account immediately",
        "your account has been compromised reset your password",
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
    ] * 5

    ham = [
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
        'text': spam + ham,
        'label': [1]*len(spam) + [0]*len(ham)
    })
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df['clean'] = df['text'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(df['clean'])
    y = df['label']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    calibrated_rf.fit(X, y)
    
    st.success("✅ Model ready!")
    return calibrated_rf, vectorizer

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def predict_spam(email, model, vectorizer):
    if is_ham_override(email):
        return {'prediction': 'ham', 'confidence': 0.99}
    
    clean = clean_text(email)
    if len(clean) < 2:
        return {'prediction': 'ham', 'confidence': 0.90}
    
    vectorized = vectorizer.transform([clean])
    pred = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    
    return {
        'prediction': 'spam' if pred == 1 else 'ham',
        'confidence': float(max(proba))
    }

# Load model
model, vectorizer = get_model()

st.set_page_config(page_title="Spam Detection", page_icon="📧", layout="wide")

st.title("📧 Spam Detection API")
st.markdown("### Powered by Random Forest")

with st.sidebar:
    st.subheader("📊 Status")
    st.success("✅ Online")

st.subheader("🔍 Test")

email_input = st.text_area("Enter email:", height=100)

if st.button("🔍 Predict", type="primary"):
    if email_input:
        result = predict_spam(email_input, model, vectorizer)
        st.json(result)
        if result["prediction"] == "spam":
            st.error(f"⚠️ SPAM ({result['confidence']:.2%})")
        else:
            st.success(f"✅ HAM ({result['confidence']:.2%})")

st.markdown("---")
st.caption("🚀 Deployed on Streamlit Cloud")
