# Melanoma Detection AI

An intelligent skin cancer detection system using deep learning to classify melanoma from dermoscopic images. Built with YOLOv8, FastAPI, and React.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.0-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)

## Overview



https://github.com/user-attachments/assets/9822679a-0720-4749-afcb-66ca36201867



This project implements an end-to-end medical AI system for melanoma detection, achieving >95% accuracy on validation data. The system provides:

- **Real-time classification** of skin lesions as melanoma or benign
- **Visual explanations** through heatmap generation
- **ABCDE criteria analysis** following dermatological standards
- **Confidence scoring** for clinical decision support
- **Beautiful, responsive web interface** with drag-and-drop functionality

## Architecture

### Model
- **Architecture**: YOLOv8 Classification (transfer learning)
- **Framework**: PyTorch
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (melanoma/normal) with confidence scores

### Backend
- **API Framework**: FastAPI (Python 3.8+)
- **ML Inference**: Ultralytics YOLO, PyTorch
- **Image Processing**: OpenCV, Pillow
- **Features**: RESTful API, real-time prediction, heatmap generation

### Frontend
- **Framework**: React 18
- **Animations**: Framer Motion
- **File Upload**: React Dropzone (drag-and-drop)
- **Styling**: Custom CSS3 with responsive design

## Model Performance

- **Accuracy**: >95% on validation set
- **Training Dataset**: 87 images (61 melanoma, 26 normal)
- **Validation Dataset**: 19 images (12 melanoma, 7 normal)
- **Inference Time**: <1 second per image
- **Model Size**: ~25MB

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- pip and npm

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ObjectDetection
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

### Running the Application

**Option 1: Automated startup (Windows)**
```bash
start_all.bat
```

**Option 2: Manual startup**

Terminal 1 (Backend):
```bash
cd backend
python app.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm start
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎓 Training the Model

The training script uses transfer learning with data augmentation to achieve high accuracy:

```bash
python train_melanoma_classifier.py
```

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 → 0.00001 (scheduled)
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Augmentation**: Rotation, flips, HSV, mixup

The trained model will be saved to `runs/melanoma_classify/melanoma_model/weights/best.pt`

## API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Melanoma Detection API is running",
  "model_loaded": true
}
```

#### `POST /predict`
Analyze an uploaded image for melanoma detection.

**Request:** Multipart form data with image file

**Response:**
```json
{
  "prediction": "melanoma",
  "confidence": 0.96,
  "explanation": {
    "description": "Potential melanoma detected",
    "details": [...],
    "recommendation": "Please consult a dermatologist...",
    "abcd_criteria": {...}
  },
  "images": {
    "original": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,..."
  }
}
```

#### `GET /model/info`
Get information about the loaded model.

## Features

### User Interface
- Drag-and-drop image upload
- Real-time preview
- Smooth animations and transitions
- Mobile-responsive design
- Professional medical aesthetic

### Medical Features
- Binary classification (melanoma/normal)
- Confidence scoring
- Visual explanations (heatmap)
- ABCDE criteria assessment:
  - **A**symmetry
  - **B**order irregularity
  - **C**olor variation
  - **D**iameter >6mm
  - **E**volving characteristics
- Medical recommendations

## Technical Details

### Model Training Approach

1. **Transfer Learning**: Started from YOLOv8 pretrained on ImageNet
2. **Data Augmentation**: Applied extensive augmentation to overcome limited dataset
3. **Regularization**: Used weight decay and early stopping to prevent overfitting
4. **Validation**: Separate validation set for unbiased accuracy metrics

### Data Augmentation Pipeline
- HSV color augmentation (H: ±1.5%, S: ±70%, V: ±40%)
- Random rotation (±20°)
- Random translation (±10%)
- Random scaling (±50%)
- Horizontal and vertical flips
- Mixup augmentation (10%)

### Explainability
- Heatmap generation using edge detection and feature highlighting
- ABCDE dermatological criteria analysis
- Confidence calibration for reliable predictions

## Development

### Project Structure
```
ObjectDetection/
├── backend/
│   └── app.py                  # FastAPI application
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.js              # Main React component
│       └── App.css             # Styling
├── classification_dataset/     # Training data
│   ├── train/
│   └── val/
├── train_melanoma_classifier.py  # Training script
├── requirements.txt            # Python dependencies
└── README.md
```

### Testing
```bash
python test_system.py
```
