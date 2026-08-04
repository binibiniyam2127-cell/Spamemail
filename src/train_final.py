"""
Final Training Script for Spam Detection
With better data handling and validation
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔥 FINAL SPAM DETECTION TRAINING")
print("="*60)

# Load data
df = pd.read_csv('data/processed/enron_spam_processed.csv')
print(f"\n📊 Loaded {len(df)} emails")
print(f"   Ham: {len(df[df['label']==0])}")
print(f"   Spam: {len(df[df['label']==1])}")

# Remove short emails (less than 10 characters) to reduce noise
df = df[df['text'].str.len() > 10]
print(f"\n📊 After filtering short emails: {len(df)} samples")

# Prepare features
X = df['processed_text']
y = df['label']

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split:")
print(f"   Training: {len(X_train)} emails")
print(f"   Testing: {len(X_test)} emails")

# Vectorize with better parameters
print("\n🔧 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✅ Feature matrix: {X_train_vec.shape}")

# Train Random Forest with better parameters
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train_vec, y_train)

# Calibrate
print("\n🎯 Calibrating with Isotonic...")
calibrated_rf = CalibratedClassifierCV(
    rf, 
    cv=5,
    method='isotonic'
)

calibrated_rf.fit(X_train_vec, y_train)

# Evaluate
y_pred = calibrated_rf.predict(X_test_vec)
y_proba = calibrated_rf.predict_proba(X_test_vec)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Cross-validation
print("\n🔍 Cross-Validation Score:")
cv_scores = cross_val_score(rf, X_train_vec, y_train, cv=5)
print(f"   CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Confidence analysis
print("\n📊 Confidence Statistics:")
print(f"   Mean Confidence: {y_proba.mean():.4f}")
print(f"   Std Confidence: {y_proba.std():.4f}")
print(f"   Min Confidence: {y_proba.min():.4f}")
print(f"   Max Confidence: {y_proba.max():.4f}")

# Confidence by class
print(f"\n📊 Confidence by Class:")
print(f"   Spam Mean: {y_proba[y_test==1].mean():.4f}")
print(f"   Ham Mean: {1 - y_proba[y_test==0].mean():.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Confusion Matrix:")
print(f"   True Ham:  {cm[0][0]:6d}  |  False Spam: {cm[0][1]:6d}")
print(f"   False Ham: {cm[1][0]:6d}  |  True Spam:  {cm[1][1]:6d}")

# Save model
print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)
joblib.dump(calibrated_rf, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Model saved!")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
