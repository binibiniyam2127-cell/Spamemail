"""
Train model on Streamlit Cloud
This runs when the app starts and model files are not found
"""
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import re
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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

def train_and_save():
    print("🚀 Training model...")
    
    # Sample data for training (using Enron dataset)
    data = {
        'text': [
            # Spam examples
            "Congratulations! You've won $1,000,000! Click here to claim your prize NOW!",
            "URGENT: Your account has been compromised. Verify immediately.",
            "FREE iPhone 15! Limited time offer. Click here to get yours!",
            "You have been selected as the winner of our lottery. Claim your cash prize!",
            "Make $5000 per day working from home! No experience needed!",
            "Your account has been locked. Click here to unlock it now.",
            "100% guaranteed weight loss! Order now and get 50% off!",
            
            # Ham examples
            "Hi John, can we meet tomorrow at 2pm to discuss the project?",
            "Thanks for your email. I'll review the report and get back to you.",
            "Please find attached the meeting agenda for next week.",
            "Good morning team, here's the updated project timeline.",
            "Dear manager, I'd like to request a vacation day for next Friday.",
            "The quarterly report is ready for review. Let me know your thoughts.",
            "Reminder: Staff meeting tomorrow at 10am in Conference Room B.",
        ],
        'label': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(data)
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
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
