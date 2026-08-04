"""
Simple training script for Streamlit Cloud
"""
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import re
import nltk

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_save():
    print("🚀 Training model...")
    
    # Create sample dataset
    spam_emails = [
        "Congratulations you won a prize",
        "Free money click here",
        "Urgent action required verify your account",
        "You have been selected for a special offer",
        "Claim your cash prize now",
        "Limited time offer act now",
        "Winner winner chicken dinner",
        "Click here to claim your reward",
        "Your account has been compromised",
        "Verify your identity immediately",
        "Free gift card waiting for you",
        "You are the lucky winner",
        "Exclusive deal just for you",
        "Don't miss this opportunity",
        "Cash bonus available now"
    ]
    
    ham_emails = [
        "Hi how are you doing today",
        "Can we meet tomorrow to discuss the project",
        "Thanks for your email I will review it",
        "Please find attached the report",
        "Good morning team here is the update",
        "Let me know your thoughts on this",
        "Looking forward to our meeting",
        "Have a great day",
        "Thanks for your help",
        "I appreciate your response",
        "Can you send me the file",
        "Let's schedule a call",
        "Thank you for your time",
        "Best regards",
        "Have a wonderful weekend"
    ]
    
    # Create DataFrame
    texts = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Clean text
    df['clean'] = df['text'].apply(clean_text)
    
    print(f"📊 Training on {len(df)} samples")
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(df['clean'])
    y = df['label']
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )
    
    rf.fit(X, y)
    
    # Calibrate
    calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    calibrated_rf.fit(X, y)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(calibrated_rf, 'models/classifier.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    print("✅ Model trained and saved!")
    return calibrated_rf, vectorizer

if __name__ == "__main__":
    train_and_save()
