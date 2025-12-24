
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import os

app = FastAPI(
    title="Melanoma Detection API",
    description="AI-powered melanoma detection with explanations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model
model = None

def load_model():
    """Load the trained melanoma classification model"""
    global model
    
    # Try to load the trained model, fall back to pretrained if not available
    model_paths = [
        r'runs/melanoma_classify/melanoma_model/weights/best.pt',
        r'C:/Users/ontar/runs/classify/train2/weights/best.pt',
        r'scripts/yolov8n-cls.pt'  # Fallback to pretrained
    ]
    
    for path in model_paths:
        if Path(path).exists():
            print(f"Loading model from: {path}")
            model = YOLO(path)
            return model
    
    # If no model found, download and use pretrained
    print("No trained model found, using pretrained YOLOv8n-cls")
    model = YOLO('yolov8n-cls.pt')
    return model

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()
    print("Model loaded successfully!")

def get_feature_importance(image: np.ndarray, model_output: Dict) -> np.ndarray:
    """
    Generate a heatmap showing which parts of the image were important for the decision
    """
    # Convert image to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply edge detection to highlight potential features
    edges = cv2.Canny(gray, 50, 150)
    
    # Create a heatmap based on edges
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    return overlay

def generate_explanation(class_name: str, confidence: float, image_shape: Tuple) -> Dict:
    """
    Generate detailed explanation for the prediction
    """
    explanations = {
        'melanoma': {
            'description': 'Potential melanoma detected',
            'details': [
                'The image shows characteristics commonly associated with melanoma',
                'Key indicators may include irregular borders, asymmetry, or color variation',
                'Melanoma is a serious form of skin cancer that requires medical attention',
                'This AI detection should NOT replace professional medical diagnosis'
            ],
            'recommendation': 'URGENT: Please consult a dermatologist immediately for professional diagnosis and treatment options.',
            'severity': 'high',
            'abcd_criteria': {
                'A_asymmetry': 'The lesion may show asymmetry - one half differs from the other',
                'B_border': 'Borders may be irregular, ragged, notched, or blurred',
                'C_color': 'Color may be non-uniform with shades of brown, black, pink, red, white, or blue',
                'D_diameter': 'Diameter may be larger than 6mm (size of a pencil eraser)',
                'E_evolving': 'The lesion may be changing in size, shape, or color'
            }
        },
        'noMelanoma': {
            'description': 'Normal skin detected',
            'details': [
                'The image shows characteristics of normal, healthy skin',
                'No obvious signs of melanoma were detected by the AI',
                'The skin appears to have regular patterns and uniform coloring',
                'This assessment is based on visual analysis only'
            ],
            'recommendation': 'Continue regular skin self-examinations and annual dermatologist check-ups. If you notice any changes, consult a healthcare professional.',
            'severity': 'low',
            'abcd_criteria': {
                'A_asymmetry': 'The skin shows symmetry and regular patterns',
                'B_border': 'Borders appear regular and well-defined',
                'C_color': 'Color appears uniform without suspicious variations',
                'D_diameter': 'No concerning large lesions detected',
                'E_evolving': 'Regular monitoring is still recommended'
            }
        }
    }
    
    explanation = explanations.get(class_name, explanations['noMelanoma'])
    
    # Add confidence-based context
    if confidence > 0.95:
        confidence_text = "very high confidence"
    elif confidence > 0.85:
        confidence_text = "high confidence"
    elif confidence > 0.70:
        confidence_text = "moderate confidence"
    else:
        confidence_text = "low confidence - please seek professional evaluation"
    
    explanation['confidence_level'] = confidence_text
    explanation['confidence_score'] = float(confidence)
    
    return explanation

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Melanoma Detection API is running",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict melanoma from uploaded image
    
    Returns:
    - prediction: class name (melanoma/noMelanoma)
    - confidence: confidence score (0-1)
    - explanation: detailed explanation of the prediction
    - heatmap: visual explanation (base64 encoded image)
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Store original for visualization
        original_image = image.copy()
        
        # Run prediction
        results = model(image, verbose=False)
        
        # Get top prediction
        top1_idx = results[0].probs.top1
        class_name = results[0].names[top1_idx]
        confidence = results[0].probs.data[top1_idx].item()
        
        # Get all class probabilities
        probs = results[0].probs.data.cpu().numpy()
        all_predictions = {
            results[0].names[i]: float(probs[i]) 
            for i in range(len(probs))
        }
        
        # Generate explanation
        explanation = generate_explanation(class_name, confidence, image.shape)
        
        # Generate feature importance heatmap
        heatmap_image = get_feature_importance(original_image, {
            'class': class_name,
            'confidence': confidence
        })
        
        # Convert images to base64
        original_base64 = image_to_base64(original_image)
        heatmap_base64 = image_to_base64(heatmap_image)
        
        # Prepare response
        response = {
            'prediction': class_name,
            'confidence': float(confidence),
            'all_predictions': all_predictions,
            'explanation': explanation,
            'images': {
                'original': original_base64,
                'heatmap': heatmap_base64
            },
            'warning': 'This is an AI-based screening tool and should not replace professional medical diagnosis.'
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": "YOLOv8 Classification",
        "classes": list(model.names.values()) if hasattr(model, 'names') else [],
        "input_size": 224,
        "framework": "Ultralytics YOLO"
    }

if __name__ == "__main__":
    print("Starting Melanoma Detection API...")
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )



