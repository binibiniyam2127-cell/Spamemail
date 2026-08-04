"""
Train model - Fixed vectorizer consistency
"""
import os
import pandas as pd
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import urllib.request

def download_enron():
    """Download Enron spam dataset"""
    url = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv"
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists('data/enron_spam_data.csv'):
        print("📥 Downloading Enron dataset...")
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
    print("🚀 Training model with Enron dataset...")
    
    # Download dataset
    download_enron()
    
    try:
        # Load dataset
        df = pd.read_csv('data/enron_spam_data.csv')
        print(f"📊 Loaded {len(df)} emails")
        
        # Combine subject and message
        df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
        df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
        
        # Remove empty or short texts
        df = df[df['text'].str.len() > 20]
        print(f"📊 After filtering: {len(df)} emails")
        
        # Clean text
        df['clean'] = df['text'].apply(clean_text)
        
        # Remove empty after cleaning
        df = df[df['clean'].str.len() > 5]
        print(f"📊 After cleaning: {len(df)} emails")
        
        # Print class distribution
        ham_count = len(df[df['label'] == 0])
        spam_count = len(df[df['label'] == 1])
        print(f"   Ham: {ham_count} emails")
        print(f"   Spam: {spam_count} emails")
        
        # Vectorize - SAME PARAMETERS EVERY TIME!
        print("🔧 Creating features...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85
        )
        
        X = vectorizer.fit_transform(df['clean'])
        y = df['label']
        print(f"✅ Feature matrix: {X.shape}")
        print(f"✅ Features: {X.shape[1]} features")
        
        # Train model
        print("🌲 Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        
        # Evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"✅ Accuracy: {accuracy:.4f}")
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf, 'models/classifier.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        # Save feature count for debugging
        with open('models/feature_count.txt', 'w') as f:
            f.write(f"Features: {X.shape[1]}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Samples: {len(df)}\n")
        
        print("✅ Model trained and saved!")
        print(f"📊 Final accuracy: {accuracy:.4f}")
        print(f"📊 Features: {X.shape[1]}")
        
        return rf, vectorizer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    train_and_save()
