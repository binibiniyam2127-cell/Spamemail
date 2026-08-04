"""
Calibrate model for better confidence scores
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

print("="*60)
print("🎯 CALIBRATING MODEL FOR BETTER CONFIDENCE")
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
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train_vec, y_train)

# Calibrate the model
print("\n🎯 Calibrating model for better confidence...")
calibrated_rf = CalibratedClassifierCV(
    rf, 
    cv=5,
    method='sigmoid'
)

calibrated_rf.fit(X_train_vec, y_train)

# Evaluate
y_pred = calibrated_rf.predict(X_test_vec)
y_proba = calibrated_rf.predict_proba(X_test_vec)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confidence analysis
print("\n📊 Confidence Statistics (Calibrated):")
print(f"   Mean Confidence: {y_proba.mean():.4f}")
print(f"   Std Confidence: {y_proba.std():.4f}")
print(f"   Min Confidence: {y_proba.min():.4f}")
print(f"   Max Confidence: {y_proba.max():.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Confusion Matrix:")
print(f"   True Ham:  {cm[0][0]:6d}  |  False Spam: {cm[0][1]:6d}")
print(f"   False Ham: {cm[1][0]:6d}  |  True Spam:  {cm[1][1]:6d}")

# Save calibrated model
print("\n💾 Saving calibrated model...")
os.makedirs('models', exist_ok=True)
joblib.dump(calibrated_rf, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Calibrated model saved!")

# Save metrics
with open('models/calibrated_metrics.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("CALIBRATED RANDOM FOREST\n")
    f.write("="*60 + "\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Mean Confidence: {y_proba.mean():.4f}\n")
    f.write(f"Std Confidence: {y_proba.std():.4f}\n")
    f.write(f"Min Confidence: {y_proba.min():.4f}\n")
    f.write(f"Max Confidence: {y_proba.max():.4f}\n")

print("💾 Saved metrics to models/calibrated_metrics.txt")

print("\n" + "="*60)
print("✅ CALIBRATION COMPLETE!")
print("="*60)
