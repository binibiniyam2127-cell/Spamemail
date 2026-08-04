"""
FastAPI Spam Detection Server
Uses updated predict.py with rule-based overrides
"""
import sys
import os
import re
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the updated predict function
from src.predict import predict_spam

app = FastAPI(
    title="Spam Detection API",
    description="ML-powered spam detection with rule-based overrides",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class EmailRequest(BaseModel):
    email: str
    id: Optional[str] = None

class BatchEmailRequest(BaseModel):
    emails: List[str]

class PredictionResponse(BaseModel):
    id: Optional[str] = None
    prediction: str
    spam_probability: float
    ham_probability: float
    confidence: float
    model_used: str
    overridden: bool
    timestamp: str

class BatchPredictionResponse(BaseModel):
    total: int
    spam_count: int
    ham_count: int
    results: List[PredictionResponse]

# Health check
@app.get("/")
async def root():
    return {
        "message": "Spam Detection API v2.0",
        "status": "online",
        "features": ["Rule-based overrides", "ML prediction", "Batch processing"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Single prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: EmailRequest):
    """Predict if a single email is spam"""
    try:
        result = predict_spam(request.email)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            'id': request.id,
            'prediction': result['prediction'],
            'spam_probability': result.get('spam_probability', 0.0),
            'ham_probability': result.get('ham_probability', 0.0),
            'confidence': result['confidence'],
            'model_used': result.get('model_used', 'Random Forest'),
            'overridden': result.get('overridden', False),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchEmailRequest):
    """Predict spam for multiple emails"""
    if not request.emails:
        raise HTTPException(status_code=400, detail="No emails provided")
    
    results = []
    spam_count = 0
    ham_count = 0
    
    for i, email in enumerate(request.emails):
        result = predict_spam(email)
        if 'error' in result:
            results.append({
                'id': str(i),
                'prediction': 'error',
                'spam_probability': 0.0,
                'ham_probability': 0.0,
                'confidence': 0.0,
                'model_used': 'Error',
                'overridden': False,
                'timestamp': datetime.now().isoformat()
            })
        else:
            results.append({
                'id': str(i),
                'prediction': result['prediction'],
                'spam_probability': result.get('spam_probability', 0.0),
                'ham_probability': result.get('ham_probability', 0.0),
                'confidence': result['confidence'],
                'model_used': result.get('model_used', 'Random Forest'),
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

# Model info
@app.get("/model/info")
async def model_info():
    """Get model information"""
    try:
        import joblib
        model = joblib.load('models/classifier.pkl')
        return {
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, 'n_estimators', 'N/A'),
            "max_depth": getattr(model, 'max_depth', 'N/A'),
            "status": "loaded",
            "version": "2.0.0",
            "features": ["Rule-based overrides", "TF-IDF vectors", "Random Forest"]
        }
    except:
        return {
            "status": "not_loaded",
            "version": "2.0.0",
            "message": "Model not loaded"
        }

# Stats endpoint
@app.get("/stats")
async def get_stats():
    """Get model performance statistics"""
    try:
        with open('models/calibrated_isotonic_metrics.txt', 'r') as f:
            stats = f.read()
        return {
            "message": "Model performance statistics",
            "data": stats
        }
    except:
        return {
            "message": "Statistics not available",
            "data": None
        }

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🔥 FASTAPI SPAM DETECTION SERVER v2.0")
    print("="*60)
    print("📡 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("📊 Health: http://localhost:8000/health")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
