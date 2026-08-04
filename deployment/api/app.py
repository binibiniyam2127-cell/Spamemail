"""
FastAPI Spam Detection Service
Run: uvicorn deployment.api.app:app --reload
Install: pip install fastapi uvicorn pydantic python-multipart
"""
import sys
import os
import re
import json
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as e:
    print(f"❌ Missing FastAPI imports: {e}")
    print("Please install: pip install fastapi uvicorn pydantic python-multipart")
    sys.exit(1)

# Load model
import joblib
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI
app = FastAPI(
    title="Spam Detection API",
    description="Machine Learning API for spam email detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None
vectorizer = None

def load_model():
    """Load model and vectorizer"""
    global model, vectorizer
    try:
        model_path = 'models/classifier.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("✅ Model loaded successfully!")
            return True
        else:
            print(f"❌ Model not found at {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
if not load_model():
    print("⚠️ Model not loaded. Please train first: python src/train.py")

# Pydantic models
class EmailRequest(BaseModel):
    email: str
    id: Optional[str] = None

class BatchEmailRequest(BaseModel):
    emails: List[str]

class PredictionResponse(BaseModel):
    id: Optional[str] = None
    prediction: str
    probability: float
    confidence: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    total: int
    spam_count: int
    ham_count: int
    results: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Helper functions
def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\$\d+\.?\d*', 'MONEY', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_single(text, id=None):
    """Predict single email"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not text or len(text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Email text is too short (minimum 3 characters)")
    
    try:
        # Clean and vectorize
        clean = clean_text(text)
        vectorized = vectorizer.transform([clean])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = float(max(probabilities))
        
        return {
            'id': id,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'probability': confidence,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Spam Detection API",
        "version": "1.0.0",
        "status": "online" if model is not None else "offline",
        "endpoints": {
            "POST /predict": "Single email prediction",
            "POST /predict/batch": "Batch email prediction",
            "GET /health": "Health check",
            "GET /docs": "Swagger documentation",
            "GET /redoc": "ReDoc documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: EmailRequest):
    """Predict if a single email is spam"""
    result = predict_single(request.email, request.id)
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchEmailRequest):
    """Predict spam for multiple emails"""
    if not request.emails:
        raise HTTPException(status_code=400, detail="No emails provided")
    
    results = []
    spam_count = 0
    ham_count = 0
    
    for i, email in enumerate(request.emails):
        try:
            result = predict_single(email, str(i))
            results.append(result)
            if result['prediction'] == 'spam':
                spam_count += 1
            else:
                ham_count += 1
        except Exception as e:
            results.append({
                'id': str(i),
                'prediction': 'error',
                'probability': 0.0,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
    
    return {
        'total': len(request.emails),
        'spam_count': spam_count,
        'ham_count': ham_count,
        'results': results
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features": vectorizer.get_feature_names_out().shape[0] if vectorizer else 0,
        "loaded": True
    }

# Run with: uvicorn deployment.api.app:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
