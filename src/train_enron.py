"""
Train on full Enron dataset - FIXED VECTORIZER
Run: python src/train_enron.py
"""
import pandas as pd
import joblib
import re
import os
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*60)
print("🚀 TRAINING ON ENRON DATASET (33,716 EMAILS)")
print("="*60)

# Download Enron dataset
url = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv"
os.makedirs('data', exist_ok=True)

if not os.path.exists('data/enron_spam_data.csv'):
    print("📥 Downloading Enron dataset (50MB)...")
    urllib.request.urlretrieve(url, 'data/enron_spam_data.csv')
    print("✅ Download complete!")
else:
    print("✅ Dataset already exists")

# Load data
df = pd.read_csv('data/enron_spam_data.csv')
print(f"\n📊 Loaded {len(df)} emails")
print(f"   Ham: {len(df[df['Spam/Ham']=='ham'])}")
print(f"   Spam: {len(df[df['Spam/Ham']=='spam'])}")

# Clean data
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

df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
df['clean'] = df['text'].apply(clean_text)

# Remove empty
df = df[df['clean'].str.len() > 10]
print(f"\n📊 After cleaning: {len(df)} emails")

# ===== FIX: VECTORIZER WITH CONSISTENT FEATURES =====
print("\n🔧 Creating features...")
vectorizer = TfidfVectorizer(
    max_features=3000,  # FIXED: 3000 features
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85
)

X = vectorizer.fit_transform(df['clean'])
y = df['label']

print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
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
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Confusion Matrix:")
print(f"   True Ham:  {cm[0][0]:6d}  |  False Spam: {cm[0][1]:6d}")
print(f"   False Ham: {cm[1][0]:6d}  |  True Spam:  {cm[1][1]:6d}")

# Calibrate
print("\n🎯 Calibrating model...")
calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
calibrated_rf.fit(X_train, y_train)

# Save
print("\n💾 Saving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(calibrated_rf, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# Save feature info
with open('models/feature_info.txt', 'w') as f:
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Samples: {len(df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")

print("✅ Model saved to models/classifier.pkl")
print("✅ Vectorizer saved to models/vectorizer.pkl")
print(f"✅ Features: {X.shape[1]}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
