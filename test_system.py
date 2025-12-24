"""
System Test Script for Melanoma Detection
Tests the complete pipeline from model loading to prediction
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 50)
    print("Testing Package Imports...")
    print("=" * 50)
    
    packages = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} - OK")
        except ImportError:
            print(f"✗ {name} - MISSING")
            all_ok = False
    
    print()
    return all_ok

def test_gpu():
    """Test GPU availability"""
    print("=" * 50)
    print("Testing GPU Availability...")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch Version: {torch.__version__}")
            return True
        else:
            print("⚠ No GPU detected - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False
    finally:
        print()

def test_dataset():
    """Test if dataset exists and has proper structure"""
    print("=" * 50)
    print("Testing Dataset Structure...")
    print("=" * 50)
    
    dataset_path = Path('classification_dataset')
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path.absolute()}")
        return False
    
    required_dirs = [
        'train/melanoma',
        'train/noMelanoma',
        'val/melanoma',
        'val/noMelanoma'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if full_path.exists():
            num_images = len(list(full_path.glob('*.jpg'))) + \
                        len(list(full_path.glob('*.jpeg'))) + \
                        len(list(full_path.glob('*.png')))
            print(f"✓ {dir_path}: {num_images} images")
        else:
            print(f"✗ {dir_path}: NOT FOUND")
            all_ok = False
    
    print()
    return all_ok

def test_model():
    """Test if model can be loaded"""
    print("=" * 50)
    print("Testing Model Loading...")
    print("=" * 50)
    
    from ultralytics import YOLO
    
    model_paths = [
        'runs/melanoma_classify/melanoma_model/weights/best.pt',
        'C:/Users/ontar/runs/classify/train2/weights/best.pt',
        'scripts/yolov8n-cls.pt'
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            try:
                print(f"Attempting to load: {path}")
                model = YOLO(path)
                print(f"✓ Model loaded successfully from: {path}")
                
                # Get model info
                if hasattr(model, 'names'):
                    print(f"  Classes: {list(model.names.values())}")
                
                return True
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                continue
    
    if model is None:
        print("✗ No trained model found")
        print("  Run: python train_melanoma_classifier.py")
        return False
    
    print()
    return True

def test_prediction():
    """Test if model can make predictions"""
    print("=" * 50)
    print("Testing Model Prediction...")
    print("=" * 50)
    
    from ultralytics import YOLO
    
    # Find a test image
    test_image_paths = [
        'images/train/melanomaTest.jpg',
        'images/train/normalTest.jpg',
        'classification_dataset/val/melanoma',
        'classification_dataset/val/noMelanoma'
    ]
    
    test_image = None
    for path in test_image_paths:
        p = Path(path)
        if p.is_file():
            test_image = str(p)
            break
        elif p.is_dir():
            images = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png'))
            if images:
                test_image = str(images[0])
                break
    
    if not test_image:
        print("✗ No test image found")
        return False
    
    print(f"Using test image: {test_image}")
    
    # Load model
    model_paths = [
        'runs/melanoma_classify/melanoma_model/weights/best.pt',
        'C:/Users/ontar/runs/classify/train2/weights/best.pt',
        'scripts/yolov8n-cls.pt'
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            try:
                model = YOLO(path)
                break
            except:
                continue
    
    if model is None:
        print("✗ Could not load model")
        return False
    
    try:
        # Run prediction
        results = model(test_image, verbose=False)
        
        # Get prediction
        top1_idx = results[0].probs.top1
        class_name = results[0].names[top1_idx]
        confidence = results[0].probs.data[top1_idx].item()
        
        print(f"✓ Prediction successful!")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        print()
        return False

def test_backend():
    """Test if backend can start"""
    print("=" * 50)
    print("Testing Backend Components...")
    print("=" * 50)
    
    try:
        # Import backend modules
        sys.path.insert(0, str(Path('backend').absolute()))
        
        # Test imports
        from fastapi import FastAPI
        from uvicorn import Config
        
        print("✓ FastAPI imports successful")
        
        # Check if backend file exists
        backend_file = Path('backend/app.py')
        if backend_file.exists():
            print(f"✓ Backend file found: {backend_file}")
        else:
            print(f"✗ Backend file not found: {backend_file}")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        print()
        return False

def test_frontend():
    """Test if frontend files exist"""
    print("=" * 50)
    print("Testing Frontend Components...")
    print("=" * 50)
    
    frontend_files = [
        'frontend/package.json',
        'frontend/src/App.js',
        'frontend/src/App.css',
        'frontend/src/index.js',
        'frontend/public/index.html'
    ]
    
    all_ok = True
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_ok = False
    
    print()
    return all_ok

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MELANOMA DETECTION SYSTEM TEST" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    results['imports'] = test_imports()
    results['gpu'] = test_gpu()
    results['dataset'] = test_dataset()
    results['model'] = test_model()
    results['prediction'] = test_prediction()
    results['backend'] = test_backend()
    results['frontend'] = test_frontend()
    
    # Summary
    print()
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.upper():.<40} {status}")
    
    print()
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print()
        print("🎉 All tests passed! System is ready to use.")
        print()
        print("To start the system:")
        print("  1. Run: start_all.bat")
        print("  OR")
        print("  2. Backend: cd backend && python app.py")
        print("     Frontend: cd frontend && npm start")
    else:
        print()
        print("⚠ Some tests failed. Please fix the issues above.")
        
        if not results['model']:
            print()
            print("💡 To train a model: python train_melanoma_classifier.py")
    
    print()

if __name__ == '__main__':
    main()



