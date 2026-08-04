"""
Preprocessing script for Enron Spam Dataset
NO NLTK - Uses pure Python with re and string
Input: data/raw/enron_spam_data.csv
Output: data/processed/enron_spam_processed.csv
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import re
import string
import numpy as np
from datetime import datetime

class SpamPreprocessor:
    def __init__(self):
        # Custom stopwords (manually defined)
        self.stop_words = set([
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
            'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
            'between', 'both', 'but', 'by', 'could', 'did', 'do', 'does', 'doing', 'down',
            'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
            'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him',
            'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in',
            'into', 'is', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'my', 'myself',
            'nor', 'of', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
            'out', 'over', 'own', 'same', 'she', 'shed', 'shell', 'shes', 'should', 'so', 'some',
            'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
            'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this',
            'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
            'wed', 'well', 'were', 'weve', 'what', 'whats', 'when', 'whens', 'where', 'wheres',
            'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'would', 'you',
            'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
        ])
        
        # Add more custom stopwords for email
        additional_stopwords = {
            'subject', 're', 'fw', 'fwd', 'mail', 'email', 'message',
            'received', 'sent', 'from', 'to', 'cc', 'bcc', 'date',
            'hello', 'hi', 'hey', 'dear', 'sincerely', 'regards',
            'thanks', 'thank', 'please', 'sorry', 'apologize'
        }
        self.stop_words.update(additional_stopwords)
        
        # Spam indicators
        self.spam_indicators = {
            'free', 'win', 'winner', 'won', 'claim', 'prize', 'cash',
            'money', 'million', 'dollar', 'congratulation', 'congrats',
            'urgent', 'immediate', 'action', 'required', 'verify',
            'confirm', 'account', 'password', 'click', 'link', 'offer',
            'limited', 'exclusive', 'guaranteed', 'remove', 'unsubscribe',
            'cheap', 'discount', 'bonus', 'credit', 'loan', 'mortgage',
            'refinance', 'insurance', 'medication', 'pharmacy', 'viagra'
        }
        
        # Ham indicators
        self.ham_indicators = {
            'meeting', 'project', 'report', 'attachment', 'document',
            'schedule', 'appointment', 'interview', 'candidate',
            'invoice', 'payment', 'receipt', 'order', 'delivery',
            'conference', 'presentation', 'proposal', 'budget',
            'agenda', 'minutes', 'deadline', 'review', 'feedback'
        }
    
    def clean_text(self, text):
        """Clean and preprocess text without NLTK"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Remove money amounts
        text = re.sub(r'\$\d+\.?\d*', ' MONEY ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        
        # Remove punctuation (keep ! and ? for features)
        text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def simple_tokenize(self, text):
        """Simple tokenization without NLTK"""
        # Split by spaces
        tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [t for t in tokens if len(t) > 2]
        
        # Simple stemming (just remove common suffixes)
        stemmed = []
        for t in tokens:
            if t.endswith('ing'):
                t = t[:-3]
            elif t.endswith('ed'):
                t = t[:-2]
            elif t.endswith('tion'):
                t = t[:-4]
            elif t.endswith('s') and len(t) > 3:
                t = t[:-1]
            stemmed.append(t)
        
        return stemmed
    
    def extract_features(self, text, tokens):
        """Extract features for Random Forest"""
        features = {}
        
        # Basic features
        features['text_length'] = len(text)
        features['word_count'] = len(tokens)
        features['unique_words'] = len(set(tokens))
        
        # Character features
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Spam/ham word counts
        spam_count = sum(1 for word in tokens if word in self.spam_indicators)
        ham_count = sum(1 for word in tokens if word in self.ham_indicators)
        
        features['spam_word_count'] = spam_count
        features['ham_word_count'] = ham_count
        features['spam_ratio'] = spam_count / max(len(tokens), 1)
        features['spam_ham_diff'] = spam_count - ham_count
        
        # Average word length
        features['avg_word_length'] = np.mean([len(w) for w in tokens]) if tokens else 0
        
        # Count special tokens
        features['url_count'] = text.count('URL')
        features['email_count'] = text.count('EMAIL')
        features['money_count'] = text.count('MONEY')
        
        return features
    
    def process(self, df, add_features=True, save_processed=True):
        print("🔧 Starting preprocessing...")
        
        # Handle Enron dataset format
        if 'Message' in df.columns and 'Spam/Ham' in df.columns:
            print("   📧 Detected Enron dataset format")
            df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
            df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
        elif 'text' in df.columns and 'label' in df.columns:
            print("   📧 Detected standard format")
            pass
        else:
            print("   ❌ Unknown format. Available columns:", df.columns.tolist())
            return df
        
        # Remove rows with empty text
        df = df[df['text'].str.len() > 0]
        
        print("   → Cleaning text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        print("   → Tokenizing...")
        df['tokens'] = df['cleaned_text'].apply(self.simple_tokenize)
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
        
        # Remove rows with empty text after preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"✅ Preprocessing complete! {len(df)} samples remaining")
        
        if save_processed:
            self.save_processed_data(df)
        
        return df
    
    def save_processed_data(self, df):
        try:
            os.makedirs('data/processed', exist_ok=True)
            
            processed_path = 'data/processed/enron_spam_processed.csv'
            df.to_csv(processed_path, index=False)
            print(f"   💾 Saved to {processed_path}")
            
            sample_path = 'data/processed/sample_100.csv'
            df.sample(min(100, len(df))).to_csv(sample_path, index=False)
            print(f"   💾 Saved sample to {sample_path}")
            
            # Save feature info
            feature_cols = [col for col in df.columns if col not in ['text', 'label', 'cleaned_text', 'tokens', 'processed_text', 'Subject', 'Message', 'Spam/Ham', 'Date', 'Message ID']]
            with open('data/processed/feature_info.txt', 'w') as f:
                f.write(f"Preprocessed Enron Spam Dataset\n")
                f.write(f"================================\n")
                f.write(f"Total: {len(df)}\n")
                f.write(f"Ham (0): {len(df[df['label']==0])}\n")
                f.write(f"Spam (1): {len(df[df['label']==1])}\n")
                f.write(f"\nFeatures:\n")
                for col in feature_cols:
                    f.write(f"  - {col}\n")
                f.write(f"\nPreprocessed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"   💾 Saved feature info to data/processed/feature_info.txt")
            
        except Exception as e:
            print(f"   ⚠️ Error saving: {e}")

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
    print("📝 Using pure Python (no NLTK)")
    df = load_data()
    if df is not None:
        print("\n📊 Sample:")
        print(df[['text', 'label', 'word_count']].head())
        print(f"\n📊 Stats: Ham={len(df[df['label']==0])}, Spam={len(df[df['label']==1])}")
        print("\n✅ Preprocessing complete!")

# Add clean_text function for prediction
def clean_text(text):
    """Simple text cleaning for prediction"""
    if not isinstance(text, str):
        text = str(text)
    import re
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\$\d+\.?\d*', 'MONEY', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
