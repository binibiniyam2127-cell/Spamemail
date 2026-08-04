"""
Train XGBoost for higher confidence
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import os

print("="*60)
print("🚀 XGBOOST SPAM DETECTOR")
print("="*60)

# Load data
df = pd.read_csv('data/processed/enron_spam_processed.csv')
print(f"\n📊 Loaded {len(df)} emails")

# Prepare features
X = df['processed_text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train XGBoost
print("\n🚀 Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(X_train_vec, y_train)

# Evaluate
y_pred = xgb.predict(X_test_vec)
y_proba = xgb.predict_proba(X_test_vec)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f}")

print(f"\n📊 Confidence Statistics:")
print(f"   Mean Confidence: {y_proba.mean():.4f}")
print(f"   Std Confidence: {y_proba.std():.4f}")
print(f"   Min Confidence: {y_proba.min():.4f}")
print(f"   Max Confidence: {y_proba.max():.4f}")

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(xgb, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("\n✅ XGBoost model saved!")
