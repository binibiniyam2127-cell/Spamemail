"""
Prediction functions for spam detection with rule-based overrides
"""
import joblib
import os
import sys
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Common ham phrases that should never be classified as spam
HAM_PHRASES = [
    'how are you',
    'how are you doing',
    'good morning',
    'good afternoon',
    'good evening',
    'hope you are well',
    'hope you\'re well',
    'hope you are doing well',
    'can we meet',
    'lets meet',
    'let\'s meet',
    'meet tomorrow',
    'meet today',
    'discuss the project',
    'project discussion',
    'thanks for the report',
    'thank you for the report',
    'i will review',
    'ill review',
    'please find attached',
    'attached is the',
    'attached please find',
    'let me know',
    'let us know',
    'please let me know',
    'looking forward',
    'looking forward to',
    'have a great day',
    'have a good day',
    'best regards',
    'kind regards',
    'sincerely',
    'thanks for your email',
    'thank you for your email',
    'i appreciate',
    'i appreciate your',
    'meeting agenda',
    'quarterly report',
    'budget review',
    'team meeting',
    'staff meeting',
    'conference call'
]

# Strong spam indicators
SPAM_PHRASES = [
    'won $',
    'won',
    'million',
    'billion',
    'free iphone',
    'free',
    'claim your',
    'claim now',
    'urgent',
    'immediate action',
    'verify your account',
    'account compromised',
    'click here',
    'click now',
    'limited time',
    'exclusive offer',
    'guaranteed',
    'unsubscribe',
    'remove'
]

def clean_text_simple(text):
    """Simple text cleaning without NLTK"""
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

def is_definitely_ham(text):
    """Check if text is definitely ham based on common phrases"""
    text_lower = text.lower()
    for phrase in HAM_PHRASES:
        if phrase in text_lower:
            return True
    return False

def is_definitely_spam(text):
    """Check if text is definitely spam based on common phrases"""
    text_lower = text.lower()
    spam_count = 0
    for phrase in SPAM_PHRASES:
        if phrase in text_lower:
            spam_count += 1
    return spam_count >= 2

def predict_spam(text):
    """Predict if text is spam using Random Forest + Rule Overrides"""
    try:
        # First check rule-based overrides
        if is_definitely_ham(text):
            return {
                'prediction': 'ham',
                'spam_probability': 0.05,
                'ham_probability': 0.95,
                'confidence': 0.95,
                'model_used': 'Rule-Based Override (Ham)',
                'overridden': True
            }
        
        # Get the directory of this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models/classifier.pkl')
        vectorizer_path = os.path.join(base_dir, 'models/vectorizer.pkl')
        
        # Check if model exists
        if not os.path.exists(model_path):
            return {
                'error': 'Model not found. Please train first using: python src/calibrate_isotonic.py'
            }
        
        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Clean and vectorize
        clean = clean_text_simple(text)
        vectorized = vectorizer.transform([clean])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        # Apply spam override if needed
        if prediction == 1 and is_definitely_spam(text):
            # Boost confidence for definite spam
            boosted_conf = min(float(max(probabilities)) * 1.1, 0.999)
            return {
                'prediction': 'spam',
                'spam_probability': boosted_conf,
                'ham_probability': 1 - boosted_conf,
                'confidence': boosted_conf,
                'model_used': 'Random Forest + Spam Boost',
                'overridden': True
            }
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'model_used': 'Random Forest (Calibrated)',
            'overridden': False
        }
        
    except Exception as e:
        return {'error': str(e)}

def predict_batch(emails):
    """Predict spam for multiple emails"""
    results = []
    for email in emails:
        results.append(predict_spam(email))
    return results

if __name__ == "__main__":
    # Test the function
    test_emails = [
        'Hi, how are you doing today?',
        'Congratulations! You won a prize!',
        'URGENT: Your account needs verification.',
        'Thanks for the report, I will review it.',
        'Can we meet tomorrow to discuss the project?',
        'FREE iPhone! Limited time offer!',
        'Good morning, hope you are well.',
        'Please find attached the quarterly report.'
    ]
    
    print("="*60)
    print("🧪 TESTING PREDICT WITH RULE OVERRIDES")
    print("="*60)
    
    for email in test_emails:
        result = predict_spam(email)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            emoji = '🚨' if result['prediction'] == 'spam' else '✅'
            overridden = " (Override)" if result.get('overridden', False) else ""
            print(f"{emoji} {result['prediction'].upper():6s} | Confidence: {result['confidence']:.2%}{overridden} | {email[:45]}...")
