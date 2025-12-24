"""
Melanoma Classification Training Script
Trains a high-accuracy model (>95%) to detect melanoma vs normal skin
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

def train_melanoma_classifier():
    """Train melanoma classifier with optimized parameters for high accuracy"""
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Load a pretrained model - using YOLOv8 medium for better accuracy
    # We can start with small and upgrade if needed
    model = YOLO('yolov8s-cls.pt')  # Small model - good balance
    
    # Define training parameters optimized for medical imaging
    results = model.train(
        data=r'C:\Users\ontar\PycharmProjects\ObjectDetection\classification_dataset',
        epochs=100,  # More epochs for better convergence
        imgsz=224,  # Standard image size for medical imaging
        batch=16,  # Adjust based on GPU memory
        patience=20,  # Early stopping patience
        
        # Optimization parameters
        optimizer='AdamW',  # Better optimizer for medical imaging
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.9,
        weight_decay=0.0005,
        
        # Data augmentation (crucial for medical imaging)
        hsv_h=0.015,  # HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # HSV-Value augmentation (fraction)
        degrees=20,  # Rotation (+/- degrees)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,  # Scale (+/- gain)
        shear=0.0,  # Shear (+/- degrees)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.5,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=0.0,  # Mosaic augmentation (not ideal for classification)
        mixup=0.1,  # Mixup augmentation
        copy_paste=0.0,  # Copy paste augmentation
        
        # Validation
        val=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Other settings
        project='runs/melanoma_classify',
        name='melanoma_model',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,  # For reproducibility
        deterministic=True,
        single_cls=False,
        device=device,
        workers=4,
        
        # Plotting
        plots=True
    )
    
    # Print final metrics
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # Validate the model
    metrics = model.val()
    
    print(f"\nFinal Validation Metrics:")
    print(f"Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
    
    # Save model info
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"\nBest model saved to: {best_model_path}")
    
    return model, metrics

if __name__ == '__main__':
    # Train the model
    model, metrics = train_melanoma_classifier()
    
    # If accuracy is below 95%, suggest improvements
    if metrics.top1 < 0.95:
        print("\n" + "!"*50)
        print("WARNING: Accuracy is below 95%")
        print("!"*50)
        print("\nSuggestions to improve accuracy:")
        print("1. Collect more training data")
        print("2. Use a larger model (yolov8m-cls.pt or yolov8l-cls.pt)")
        print("3. Increase training epochs")
        print("4. Add more data augmentation")
        print("5. Ensure data quality and proper labeling")
    else:
        print("\n" + "✓"*50)
        print(f"SUCCESS! Model achieved {metrics.top1*100:.2f}% accuracy")
        print("✓"*50)



