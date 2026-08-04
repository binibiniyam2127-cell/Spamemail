"""
Boost confidence using temperature scaling
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import os

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

print("="*60)
print("🌡️ BOOSTING CONFIDENCE WITH TEMPERATURE SCALING")
print("="*60)

# Load model
model = joblib.load('models/classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

print("\n📊 Loading dataset for temperature scaling...")

# Download Enron data if needed
import urllib.request
if not os.path.exists('data/enron_spam_data.csv'):
    print("📥 Downloading Enron dataset...")
    url = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv"
    os.makedirs('data', exist_ok=True)
    urllib.request.urlretrieve(url, 'data/enron_spam_data.csv')
    print("✅ Download complete!")

df = pd.read_csv('data/enron_spam_data.csv')
df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
df['clean'] = df['text'].apply(clean_text)
df = df[df['clean'].str.len() > 10]

# Split for temperature scaling
X = df['clean']
y = df['label']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize
X_train_vec = vectorizer.transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Get raw probabilities
raw_probs = model.predict_proba(X_val_vec)[:, 1]

print(f"\n📊 Raw Confidence Statistics:")
print(f"   Mean: {raw_probs.mean():.4f}")
print(f"   Std: {raw_probs.std():.4f}")
print(f"   Min: {raw_probs.min():.4f}")
print(f"   Max: {raw_probs.max():.4f}")

# Find optimal temperature
from scipy.optimize import minimize

def nll_loss(temp):
    """Negative log likelihood loss"""
    if temp <= 0:
        return 1e10
    # Scale logits
    scaled = np.log(raw_probs / (1 - raw_probs + 1e-10)) / temp
    scaled_probs = 1 / (1 + np.exp(-scaled))
    # Clip to avoid log(0)
    scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
    # Negative log likelihood
    loss = -np.mean(y_val * np.log(scaled_probs) + (1 - y_val) * np.log(1 - scaled_probs))
    return loss

# Find optimal temperature
result = minimize(nll_loss, 1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
optimal_temp = result.x[0]
print(f"\n🌡️ Optimal Temperature: {optimal_temp:.4f}")

# Apply temperature scaling
scaled_probs = 1 / (1 + np.exp(-np.log(raw_probs / (1 - raw_probs + 1e-10)) / optimal_temp))
scaled_probs = np.clip(scaled_probs, 0, 1)

print(f"\n📊 After Temperature Scaling:")
print(f"   Mean: {scaled_probs.mean():.4f}")
print(f"   Std: {scaled_probs.std():.4f}")
print(f"   Min: {scaled_probs.min():.4f}")
print(f"   Max: {scaled_probs.max():.4f}")

# Create boosted model wrapper
class ConfidenceBoostedModel:
    def __init__(self, base_model, vectorizer, temperature):
        self.base_model = base_model
        self.vectorizer = vectorizer
        self.temperature = temperature
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)
        # Boost confidence
        boosted = np.zeros_like(raw_probs)
        for i in range(raw_probs.shape[0]):
            prob = raw_probs[i, 1]
            # Apply temperature scaling
            scaled = 1 / (1 + np.exp(-np.log(prob / (1 - prob + 1e-10)) / self.temperature))
            boosted[i, 0] = 1 - scaled
            boosted[i, 1] = scaled
        return boosted

# Create boosted model
boosted_model = ConfidenceBoostedModel(model, vectorizer, optimal_temp)

# Test on examples
test_emails = [
    'Congratulations! You won $1,000,000! Click here to claim!',
    'Hi, how are you doing today?',
    'URGENT: Your account has been compromised!',
    'Thanks for your email. I will review it.',
    'FREE iPhone! Limited time offer!'
]

print("\n" + "="*60)
print("🧪 TESTING BOOSTED CONFIDENCE")
print("="*60)

for email in test_emails:
    clean = clean_text(email)
    vec = vectorizer.transform([clean])
    pred = boosted_model.predict(vec)[0]
    proba = boosted_model.predict_proba(vec)[0]
    confidence = max(proba)
    
    label = 'SPAM' if pred == 1 else 'HAM'
    emoji = '🚨' if pred == 1 else '✅'
    print(f'{emoji} {label:6s} | Confidence: {confidence:.2%} | {email[:50]}...')

# Save boosted model
joblib.dump(boosted_model, 'models/classifier.pkl')
print("\n✅ Boosted model saved to models/classifier.pkl")

print("\n" + "="*60)
print("✅ BOOSTING COMPLETE!")
print("="*60)
