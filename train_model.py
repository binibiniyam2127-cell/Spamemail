import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# 50+ spam examples
spam = [
    # Spam patterns
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
    "free gift card waiting for you",
    "you are the lucky winner",
    "exclusive deal just for you",
    "don't miss this opportunity",
    "cash bonus available now",
    "win a free iphone",
    "you won a million dollars",
    "claim your free gift",
    "urgent security alert",
    "account suspended click here",
    "free vacation to bahamas",
    "you have been chosen",
    "special promotion just for you",
    "act fast limited stock",
    "guaranteed approval",
    "lowest price guaranteed",
    "free trial offer",
    "risk free opportunity",
    "make money fast",
    "work from home opportunity",
    "earn extra cash",
    "no experience needed",
    "get rich quick",
    "passive income opportunity",
    "financial freedom",
    "debt relief program",
    "credit card offer approved",
    "loan approval guaranteed",
    "mortgage refinance offer",
    "insurance quote request",
    "medication at low cost",
    "pharmacy discount",
    "health supplement offer",
    "weight loss miracle",
    "anti aging breakthrough"
]

# 50+ ham examples  
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
    "can you send me the file",
    "let's schedule a call",
    "thank you for your time",
    "best regards",
    "have a wonderful weekend",
    "see you at the meeting",
    "will send the documents shortly",
    "please review the proposal",
    "looking forward to hearing from you",
    "let's touch base tomorrow",
    "hope you are well",
    "thank you for your assistance",
    "i will get back to you soon",
    "we should catch up soon",
    "please confirm your availability",
    "looking forward to working with you",
    "have a productive week",
    "talk to you then",
    "enjoy your weekend",
    "take care of yourself",
    "keep up the good work",
    "thanks for the update",
    "i understand your concern",
    "let's find a solution together",
    "appreciate your patience",
    "will follow up next week",
    "please let me know if you have questions",
    "looking forward to your feedback",
    "hope all is well",
    "take care and stay safe"
]

df = pd.DataFrame({
    'text': spam + ham,
    'label': [1]*len(spam) + [0]*len(ham)
})

df['clean'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean'])
y = df['label']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

import joblib
import os
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print(f"✅ Model trained on {len(df)} samples")
