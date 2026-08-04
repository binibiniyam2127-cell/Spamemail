"""
Train Random Forest model for spam detection
"""
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import load_data

def train_random_forest():
    try:
        print("="*60)
        print("🌲 RANDOM FOREST SPAM DETECTOR")
        print("="*60)
        
        df = load_data('data/spam.csv')
        if df is None:
            print("❌ Failed to load data")
            return None
        
        print("\n📊 Dataset:", len(df), "emails")
        print("   Ham:", len(df[df['label']==0]))
        print("   Spam:", len(df[df['label']==1]))
        
        X = df['processed_text']
        y = df['label']
        
        print("\n🔄 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("   Training:", len(X_train), "emails")
        print("   Testing:", len(X_test), "emails")
        
        print("\n🔧 Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print("✅ Feature matrix:", X_train_vec.shape)
        
        print("\n🌲 Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train_vec, y_train)
        print("✅ Model trained!")
        
        y_pred = rf.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print("\n🎯 Accuracy:", accuracy)
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print("   True Ham: ", cm[0][0], " | False Spam:", cm[0][1])
        print("   False Ham:", cm[1][0], " | True Spam: ", cm[1][1])
        
        print("\n📊 TOP 15 IMPORTANT WORDS:")
        feature_names = vectorizer.get_feature_names_out()
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        for i, idx in enumerate(indices):
            print(f"   {i+1:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf, 'models/classifier.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        print("\n✅ Model saved to models/classifier.pkl")
        print("✅ Vectorizer saved to models/vectorizer.pkl")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE! 🎉")
        print("="*60)
        
        return rf, vectorizer, accuracy
        
    except Exception as e:
        print("\n❌ Error:", e)
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_random_forest()