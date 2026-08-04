#!/usr/bin/env python
import os
import sys
import subprocess

# Run the preprocess script with -P flag
result = subprocess.run([
    sys.executable,
    '-P',
    '-c',
    '''
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import re
import string
import numpy as np
from datetime import datetime

# Now import NLTK - should work with -P flag
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ NLTK data downloaded")

class SpamPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        self.spam_indicators = {
            'free', 'win', 'winner', 'won', 'claim', 'prize', 'cash',
            'money', 'million', 'dollar', 'congratulation', 'congrats',
            'urgent', 'immediate', 'action', 'required', 'verify',
            'confirm', 'account', 'password', 'click', 'link', 'offer'
        }
        
        self.ham_indicators = {
            'meeting', 'project', 'report', 'attachment', 'document',
            'schedule', 'appointment', 'interview', 'candidate',
            'invoice', 'payment', 'receipt', 'order', 'delivery'
        }
    
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        text = re.sub(r'\$\d+\.?\d*', 'MONEY', text)
        text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            tokens = [t for t in tokens if len(t) > 1]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            return tokens
        except:
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.stop_words]
            tokens = [t for t in tokens if len(t) > 1]
            return tokens
    
    def extract_features(self, text, tokens):
        features = {}
        features['text_length'] = len(text)
        features['word_count'] = len(tokens)
        features['unique_words'] = len(set(tokens))
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        spam_count = sum(1 for word in tokens if word in self.spam_indicators)
        ham_count = sum(1 for word in tokens if word in self.ham_indicators)
        
        features['spam_word_count'] = spam_count
        features['ham_word_count'] = ham_count
        features['spam_ratio'] = spam_count / max(len(tokens), 1)
        features['spam_ham_diff'] = spam_count - ham_count
        features['avg_word_length'] = np.mean([len(w) for w in tokens]) if tokens else 0
        features['url_count'] = text.count('URL')
        features['email_count'] = text.count('EMAIL')
        features['money_count'] = text.count('MONEY')
        
        return features
    
    def process(self, df, add_features=True, save_processed=True):
        print("🔧 Starting preprocessing...")
        
        if 'Message' in df.columns and 'Spam/Ham' in df.columns:
            print("   📧 Detected Enron dataset format")
            df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
            df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
        else:
            print("   ❌ Unknown format")
            return df
        
        print("   → Cleaning text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        print("   → Tokenizing and lemmatizing...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        if add_features:
            print("   → Extracting features...")
            feature_dicts = df.apply(
                lambda row: self.extract_features(row['cleaned_text'], row['tokens']),
                axis=1
            )
            feature_df = pd.DataFrame(feature_dicts.tolist())
            df = pd.concat([df, feature_df], axis=1)
        
        df = df.fillna(0)
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"✅ Preprocessing complete! {len(df)} samples remaining")
        
        if save_processed:
            os.makedirs('data/processed', exist_ok=True)
            df.to_csv('data/processed/enron_spam_processed.csv', index=False)
            print(f"   💾 Saved to data/processed/enron_spam_processed.csv")
            
            df.sample(min(100, len(df))).to_csv('data/processed/sample_100.csv', index=False)
            print(f"   💾 Saved sample to data/processed/sample_100.csv")
        
        return df

def load_data(filepath='data/raw/enron_spam_data.csv', add_features=True, save_processed=True):
    try:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"📊 Loaded {len(df)} emails from {filepath}")
        
        preprocessor = SpamPreprocessor()
        df = preprocessor.process(df, add_features=add_features, save_processed=save_processed)
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*60)
    print("🔧 PREPROCESSING ENRON SPAM DATASET")
    print("="*60)
    df = load_data()
    if df is not None:
        print(f"\\n📊 Stats: Ham={len(df[df['label']==0])}, Spam={len(df[df['label']==1])}")
        print("\\n✅ Preprocessing complete!")
    '''
])

sys.exit(result.returncode)
