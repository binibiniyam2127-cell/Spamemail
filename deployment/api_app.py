import streamlit as st
import joblib
import re
import os

# === RULE-BASED OVERRIDES ===
HAM_PHRASES = [
    'how are you', 'how are you doing', 'good morning', 'good afternoon',
    'hope you are well', 'can we meet', 'lets meet', 'meet tomorrow',
    'discuss the project', 'thanks for the report', 'thank you for the report',
    'i will review', 'please find attached', 'let me know', 'looking forward',
    'have a great day', 'best regards', 'kind regards', 'thanks for your email',
    'thank you for your email', 'i appreciate', 'meeting agenda', 'quarterly report',
    'budget review', 'team meeting', 'how are you today', 'hope all is well',
    'take care', 'good to hear', 'great to hear', 'thanks', 'thank you',
    'hi how are you', 'hello', 'hey', 'good morning team', 'good evening'
]

SPAM_PHRASES = [
    'won $', 'won', 'million', 'billion', 'free iphone', 'free',
    'claim your', 'claim now', 'urgent', 'verify your account',
    'account compromised', 'click here', 'click now', 'limited time',
    'exclusive offer', 'guaranteed', 'unsubscribe', 'remove'
]

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer, "loaded"
    except Exception as e:
        return None, None, f"error: {e}"

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

def is_ham(text):
    text_lower = text.lower()
    for phrase in HAM_PHRASES:
        if phrase in text_lower:
            return True
    return False

def is_spam(text):
    text_lower = text.lower()
    count = sum(1 for p in SPAM_PHRASES if p in text_lower)
    return count >= 2

def predict_spam(email, model, vectorizer):
    if model is None or vectorizer is None:
        return {'error': 'Model not loaded'}
    
    # Rule override: if it's definitely ham
    if is_ham(email):
        return {
            'prediction': 'ham',
            'confidence': 0.99,
            'spam_probability': 0.01,
            'ham_probability': 0.99,
            'overridden': True
        }
    
    clean = clean_text(email)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = float(max(probabilities))
    
    # Boost spam if strong indicators
    if prediction == 1 and is_spam(email):
        confidence = min(confidence * 1.2, 0.99)
        return {
            'prediction': 'spam',
            'confidence': confidence,
            'spam_probability': confidence,
            'ham_probability': 1 - confidence,
            'overridden': True
        }
    
    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'confidence': confidence,
        'spam_probability': float(probabilities[1]),
        'ham_probability': float(probabilities[0]),
        'overridden': False
    }

# Load model
model, vectorizer, status = load_model()

st.set_page_config(page_title="Spam Detection API", page_icon="📧", layout="wide")

st.title("📧 Spam Detection API")
st.markdown("### Powered by Random Forest + Enron Dataset (97% Accuracy)")

with st.sidebar:
    st.subheader("📊 Status")
    if model is not None:
        st.success("✅ Model Loaded")
        st.info("📊 Accuracy: 97%")
    else:
        st.error(f"❌ {status}")

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
                    if result.get('overridden', False):
                        st.info("📌 Rule override applied")
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
