"""
PHASE 6: REST API Backend using FastAPI

This script provides a REST API for Human Action Recognition.

API Endpoints:
    POST /predict - Upload image and get action prediction
    GET /health - Check API health status
    GET /classes - Get list of available action classes

Technologies:
    - FastAPI: Modern web framework
    - Uvicorn: ASGI server
    - TensorFlow: Model inference
    - PIL: Image processing

Author: Deep Learning Assignment
Date: January 2026
"""

import os
import pickle
import numpy as np
from PIL import Image
import io
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "../training/models/har_cnn_lstm.h5"
LABEL_ENCODER_PATH = "../training/models/label_encoder.pkl"
IMG_SIZE = (224, 224)

# ==========================================
# INITIALIZE FASTAPI APP
# ==========================================

app = FastAPI(
    title="Human Action Recognition API",
    description="CNN+LSTM based action recognition from images",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

model = None
label_encoder = None
class_names = None


# ==========================================
# LOAD MODEL AT STARTUP
# ==========================================

@app.on_event("startup")
async def load_model():
    """
    Load trained model and label encoder when API starts
    
    Why at startup?
        - Faster predictions (model already in memory)
        - Fail fast if model files are missing
        - Validate model integrity before serving requests
    """
    
    global model, label_encoder, class_names
    
    print("\n" + "🚀 "*30)
    print("STARTING API SERVER")
    print("🚀 "*30)
    
    try:
        # Load model
        print(f"\n📦 Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Load label encoder
        print(f"\n📦 Loading label encoder from: {LABEL_ENCODER_PATH}")
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")
        
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        class_names = label_encoder.classes_.tolist()
        print(f"✅ Label encoder loaded! Classes: {len(class_names)}")
        print(f"   {', '.join(class_names)}")
        
        print("\n✅ API Ready for predictions!")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        raise


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image for model inference
    
    Args:
        image_bytes: Raw image bytes from upload
    
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    
    Steps:
        1. Convert bytes to PIL Image
        2. Convert to RGB (handle RGBA, grayscale)
        3. Resize to model input size (224x224)
        4. Convert to numpy array
        5. Normalize to [0, 1]
        6. Add batch and sequence dimensions
    
    Why these steps?
        - RGB conversion: Model expects 3 channels
        - Resize: Match training input size
        - Normalize: Same preprocessing as training
        - Dimensions: (batch, sequence, height, width, channels)
    """
    
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMG_SIZE)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch and sequence dimensions: (1, 1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # Sequence dimension
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Human Action Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for action prediction",
            "/health": "GET - Check API health",
            "/classes": "GET - Get available action classes"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        dict: API status and model availability
    
    Why health check?
        - Monitor API availability
        - Verify model is loaded
        - Useful for deployment orchestration
    """
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "num_classes": len(class_names) if class_names else 0
    }


@app.get("/classes")
async def get_classes():
    """
    Get list of action classes
    
    Returns:
        dict: List of available action classes
    
    Why this endpoint?
        - Frontend can display available classes
        - Documentation for API users
        - Validate model predictions
    """
    
    if class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "num_classes": len(class_names),
        "classes": class_names
    }


@app.post("/predict")
async def predict_action(file: UploadFile = File(...)):
    """
    Predict action from uploaded image
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
    
    Returns:
        dict: Prediction results with action label and confidence
    
    Response Format:
        {
            "action": "dancing",
            "confidence": 0.95,
            "all_predictions": [
                {"action": "dancing", "confidence": 0.95},
                {"action": "clapping", "confidence": 0.03},
                ...
            ]
        }
    
    Why return top predictions?
        - Provides context (what else model considered)
        - Useful for debugging low confidence predictions
        - Helps understand model uncertainty
    """
    
    # Check if model is loaded
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess image
        preprocessed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all predictions (instead of just top 5)
        all_indices = np.argsort(predictions[0])[::-1]  # Sort all predictions in descending order
        all_predictions = [
            {
                "action": label_encoder.inverse_transform([idx])[0],
                "confidence": float(predictions[0][idx])
            }
            for idx in all_indices
        ]
        
        # Return results
        return {
            "success": True,
            "action": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions,
            "filename": file.filename
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Please check API documentation at /"
        }
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong. Please try again."
        }
    )


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("Starting Human Action Recognition API Server")
    print("="*80)
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("🔗 API Endpoint: http://localhost:8000\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload in production
    )
