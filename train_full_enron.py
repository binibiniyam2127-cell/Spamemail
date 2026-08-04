"""
Train model with full Enron dataset for high confidence
"""
import os
import pandas as pd
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import urllib.request

def download_enron():
    """Download Enron spam dataset"""
    url = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv"
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists('data/enron_spam_data.csv'):
        print("📥 Downloading Enron dataset (33,716 emails)...")
        urllib.request.urlretrieve(url, 'data/enron_spam_data.csv')
        print("✅ Download complete!")
    else:
        print("✅ Dataset already exists")

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
    print("="*60)
    print("🚀 TRAINING ON FULL ENRON DATASET")
    print("="*60)
    
    # Download dataset
    download_enron()
    
    # Load dataset
    print("\n📊 Loading dataset...")
    df = pd.read_csv('data/enron_spam_data.csv')
    print(f"   Total emails: {len(df)}")
    
    # Combine subject and message
    df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
    df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df['clean'] = df['text'].apply(clean_text)
    
    # Remove empty text
    df = df[df['clean'].str.len() > 10]
    print(f"   After cleaning: {len(df)} emails")
    
    # Show class distribution
    print(f"\n📊 Class Distribution:")
    print(f"   Ham: {len(df[df['label']==0])}")
    print(f"   Spam: {len(df[df['label']==1])}")
    
    # Vectorize
    print("\n🔧 Creating features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85
    )
    
    X = vectorizer.fit_transform(df['clean'])
    y = df['label']
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Get confidence on test set
    y_proba = rf.predict_proba(X_test)[:, 1]
    print(f"\n📊 Confidence Statistics:")
    print(f"   Mean Confidence: {y_proba.mean():.4f}")
    print(f"   Min Confidence: {y_proba.min():.4f}")
    print(f"   Max Confidence: {y_proba.max():.4f}")
    
    # Calibrate for better confidence
    print("\n🎯 Calibrating model...")
    calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    calibrated_rf.fit(X_train, y_train)
    
    # Re-evaluate calibrated
    y_pred_cal = calibrated_rf.predict(X_test)
    accuracy_cal = (y_pred_cal == y_test).mean()
    y_proba_cal = calibrated_rf.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 After Calibration:")
    print(f"   Accuracy: {accuracy_cal:.4f}")
    print(f"   Mean Confidence: {y_proba_cal.mean():.4f}")
    print(f"   Min Confidence: {y_proba_cal.min():.4f}")
    print(f"   Max Confidence: {y_proba_cal.max():.4f}")
    
    # Save models
    print("\n💾 Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(calibrated_rf, 'models/classifier.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    print("✅ Model saved to models/classifier.pkl")
    print("✅ Vectorizer saved to models/vectorizer.pkl")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    return calibrated_rf, vectorizer

if __name__ == "__main__":
    train_and_save()
