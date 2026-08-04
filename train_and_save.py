"""
Train and save both model and vectorizer
"""
import pandas as pd
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("🚀 Training model and saving vectorizer...")

# Sample data (since we can't use full Enron here)
spam = [
    "congratulations you won a prize",
    "free money click here",
    "urgent action required verify your account",
    "you have been selected for a special offer",
    "claim your cash prize now",
    "limited time offer act now",
    "winner winner chicken dinner",
    "click here to claim your reward",
    "your account has been compromised",
    "verify your identity immediately",
]

ham = [
    "hi how are you doing today",
    "can we meet tomorrow to discuss the project",
    "thanks for your email i will review it",
    "please find attached the report",
    "good morning team here is the update",
    "let me know your thoughts on this",
    "looking forward to our meeting",
    "have a great day",
    "thanks for your help",
    "i appreciate your response",
]

df = pd.DataFrame({
    'text': spam + ham,
    'label': [1]*len(spam) + [0]*len(ham)
})

df['clean'] = df['text'].apply(clean_text)

# Create and save vectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['clean'])
y = df['label']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save both
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")
print(f"✅ Model: models/classifier.pkl")
print(f"✅ Vectorizer: models/vectorizer.pkl")
print(f"✅ Features: {X.shape[1]}")
