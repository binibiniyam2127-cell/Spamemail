"""
Utility functions for spam detection
"""
import pandas as pd
import joblib
import os

def get_dataset_stats(filepath='data/spam.csv'):
    """Get statistics about the dataset"""
    try:
        if not os.path.exists(filepath):
            return {'error': 'Dataset not found'}
        
        df = pd.read_csv(filepath)
        total = len(df)
        spam_count = len(df[df['label'] == 1])
        ham_count = len(df[df['label'] == 0])
        
        return {
            'total': total,
            'spam': spam_count,
            'ham': ham_count,
            'spam_percentage': (spam_count / total) * 100 if total > 0 else 0,
            'ham_percentage': (ham_count / total) * 100 if total > 0 else 0
        }
    except Exception as e:
        return {'error': str(e)}

def get_model_info():
    """Get information about the trained model"""
    try:
        if not os.path.exists('models/classifier.pkl'):
            return {'error': 'Model not found'}
        
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        
        params = model.get_params()
        
        return {
            'model_type': type(model).__name__,
            'n_estimators': params.get('n_estimators', 'N/A'),
            'max_depth': params.get('max_depth', 'N/A'),
            'vectorizer': type(vectorizer).__name__,
            'max_features': vectorizer.get_params().get('max_features', 'N/A')
        }
    except Exception as e:
        return {'error': str(e)}
