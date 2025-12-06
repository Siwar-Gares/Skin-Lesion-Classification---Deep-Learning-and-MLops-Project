"""
FastAPI backend for skin lesion classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from typing import Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import uvicorn

from src.models.architecture import create_model, load_checkpoint
from src.data.dataset import CLASS_NAMES


# Initialize FastAPI app
app = FastAPI(
    title="Skin Lesion Classification API",
    description="Deep Learning API for classifying skin lesions using EfficientNet-B3",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None
transform = None


def get_transform(image_size: int = 224):
    """Get inference transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device, transform
    
    print("Loading model...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Create model
        model = create_model(
            architecture='efficientnet_b3',
            num_classes=7,
            pretrained=False,
            device=device
        )
        
        # Load checkpoint
        checkpoint_path = 'checkpoints/best_model.pth'
        model = load_checkpoint(model, checkpoint_path, device=device)
        model.eval()
        
        # Initialize transform
        transform = get_transform(image_size=224)
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Skin Lesion Classification API",
        "status": "running",
        "model": "EfficientNet-B3",
        "classes": CLASS_NAMES
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/classes")
async def get_classes():
    """Get list of supported classes"""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin lesion class from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        image_np = np.array(image)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_class = probabilities.max(1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            # Get all class probabilities
            class_probs = probabilities[0].cpu().numpy()
        
        # Prepare response
        predictions = []
        for idx, prob in enumerate(class_probs):
            predictions.append({
                "class": CLASS_NAMES[idx],
                "probability": float(prob),
                "percentage": float(prob * 100)
            })
        
        # Sort by probability
        predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
        
        return {
            "success": True,
            "predicted_class": CLASS_NAMES[predicted_class],
            "confidence": float(confidence),
            "confidence_percentage": float(confidence * 100),
            "all_predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict multiple images in batch
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Read and preprocess image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = np.array(image)
            transformed = transform(image=image_np)
            image_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = probabilities.max(1)
                predicted_class = predicted_class.item()
                confidence = confidence.item()
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": CLASS_NAMES[predicted_class],
                "confidence": float(confidence),
                "confidence_percentage": float(confidence * 100)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "architecture": model.architecture,
        "num_classes": model.num_classes,
        "total_parameters": model.get_num_params(),
        "trainable_parameters": model.get_trainable_params(),
        "device": str(device),
        "input_size": [224, 224],
        "classes": CLASS_NAMES
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
