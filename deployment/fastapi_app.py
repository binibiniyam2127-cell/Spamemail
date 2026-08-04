"""
FastAPI Spam Detection Service with Rule Overrides
"""
import os
import sys
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Spam Detection API",
    description="ML-powered spam detection with rule overrides",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    email: str
    id: Optional[str] = None

class BatchEmailRequest(BaseModel):
    emails: List[str]

model = None
vectorizer = None

# Rule-based overrides
HAM_PHRASES = [
    'how are you', 'how are you doing', 'good morning',
    'hope you are well', 'can we meet', 'meet tomorrow',
    'discuss the project', 'thanks for the report',
    'thank you for your email', 'please find attached',
    'looking forward to', 'best regards', 'kind regards',
    'have a great day', 'let me know', 'i will review'
]

def is_ham(text):
    """Check if text is definitely ham"""
    text_lower = text.lower()
    for phrase in HAM_PHRASES:
        if phrase in text_lower:
            return True
    return False

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

def load_model():
    global model, vectorizer
    try:
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_spam(text):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Check rule-based override first
        if is_ham(text):
            return {
                'prediction': 'ham',
                'confidence': 0.95,
                'spam_probability': 0.05,
                'ham_probability': 0.95,
                'overridden': True
            }
        
        clean = clean_text(text)
        vectorized = vectorizer.transform([clean])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(max(probabilities)),
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0]),
            'overridden': False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

load_model()

@app.get("/")
async def root():
    return {
        "message": "Spam Detection API",
        "version": "2.0.0",
        "status": "online" if model is not None else "offline"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: EmailRequest):
    result = predict_spam(request.email)
    return {
        'id': request.id,
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'spam_probability': result['spam_probability'],
        'ham_probability': result['ham_probability'],
        'overridden': result.get('overridden', False),
        'timestamp': datetime.now().isoformat()
    }

@app.post("/predict/batch")
async def predict_batch(request: BatchEmailRequest):
    if not request.emails:
        raise HTTPException(status_code=400, detail="No emails provided")
    
    results = []
    spam_count = 0
    ham_count = 0
    
    for i, email in enumerate(request.emails):
        result = predict_spam(email)
        results.append({
            'id': str(i),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'spam_probability': result['spam_probability'],
            'ham_probability': result['ham_probability'],
            'overridden': result.get('overridden', False),
            'timestamp': datetime.now().isoformat()
        })
        
        if result['prediction'] == 'spam':
            spam_count += 1
        else:
            ham_count += 1
    
    return {
        'total': len(request.emails),
        'spam_count': spam_count,
        'ham_count': ham_count,
        'results': results
    }

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": vectorizer.get_feature_names_out().shape[0] if vectorizer else 0,
        "status": "loaded",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
