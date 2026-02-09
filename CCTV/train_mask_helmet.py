"""
Train YOLOv8 model for mask and helmet detection
Fine-tunes pretrained YOLOv8 on custom dataset
Usage: python train_mask_helmet.py
"""

from ultralytics import YOLO
import torch
import os
import yaml
from pathlib import Path


def check_dataset(yaml_path):
    """Verify dataset configuration and paths"""
    print("Checking dataset configuration...")
    
    if not os.path.exists(yaml_path):
        print(f"Error: Dataset config not found: {yaml_path}")
        return False
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['path', 'train', 'val', 'names', 'nc']
    for field in required_fields:
        if field not in config:
            print(f"Error: Missing required field '{field}' in {yaml_path}")
            return False
    
    # Check paths exist
    base_path = config['path']
    train_path = os.path.join(base_path, config['train'])
    val_path = os.path.join(base_path, config['val'])
    
    if not os.path.exists(train_path):
        print(f"Error: Training images not found: {train_path}")
        print("Hint: Run python scripts/split_dataset.py first")
        return False
    
    if not os.path.exists(val_path):
        print(f"Error: Validation images not found: {val_path}")
        return False
    
    # Count images
    train_images = len(list(Path(train_path).glob("*.jpg"))) + len(list(Path(train_path).glob("*.png")))
    val_images = len(list(Path(val_path).glob("*.jpg"))) + len(list(Path(val_path).glob("*.png")))
    
    print(f"✓ Dataset config valid")
    print(f"✓ Training images: {train_images}")
    print(f"✓ Validation images: {val_images}")
    print(f"✓ Classes: {config['names']}\n")
    
    if train_images < 100:
        print(f"Warning: Only {train_images} training images. Recommend at least 100+ for good results")
    
    return True


def train_model(
    data_yaml='data/mask_dataset.yaml',
    model_size='n',  # n=nano, s=small, m=medium, l=large, x=xlarge
    epochs=100,
    batch=16,
    imgsz=640,
    device=None,
    resume=False
):
    """
    Train mask/helmet detection model
    
    Args:
        data_yaml: Path to dataset configuration
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch: Batch size (reduce if GPU memory error)
        imgsz: Image size for training
        device: Device to use (None=auto, 0=GPU, 'cpu'=CPU)
        resume: Resume from last checkpoint
    """
    
    # Check GPU availability
    print("="*60)
    print("SYSTEM CHECK")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Training on CPU (will be slower)")
    print()
    
    # Validate dataset
    if not check_dataset(data_yaml):
        return
    
    # Auto-detect device if not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Load pretrained model
    model_path = f'yolov8{model_size}.pt'
    print(f"Loading pretrained model: {model_path}")
    model = YOLO(model_path)
    
    # Training configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")
    print(f"Resume: {resume}")
    print("="*60 + "\n")
    
    # Start training
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=4,
            patience=20,              # Early stopping patience
            save=True,                # Save checkpoints
            save_period=10,           # Save every N epochs
            cache=False,              # Cache images (use True if enough RAM)
            
            # Output directories
            project='runs/mask_helmet',
            name='train_v1',
            exist_ok=True,
            
            # Augmentation settings
            hsv_h=0.015,              # Hue augmentation
            hsv_s=0.7,                # Saturation
            hsv_v=0.4,                # Value/brightness
            degrees=10.0,             # Rotation
            translate=0.1,            # Translation
            scale=0.5,                # Scaling
            shear=0.0,                # Shear
            perspective=0.0,          # Perspective
            flipud=0.0,               # Vertical flip
            fliplr=0.5,               # Horizontal flip
            mosaic=1.0,               # Mosaic augmentation
            mixup=0.0,                # Mixup augmentation
            copy_paste=0.0,           # Copy-paste augmentation
            
            # Hyperparameters
            lr0=0.01,                 # Initial learning rate
            lrf=0.01,                 # Final learning rate
            momentum=0.937,           # Momentum
            weight_decay=0.0005,      # Weight decay
            warmup_epochs=3.0,        # Warmup epochs
            warmup_momentum=0.8,      # Warmup momentum
            box=7.5,                  # Box loss gain
            cls=0.5,                  # Class loss gain
            dfl=1.5,                  # DFL loss gain
            
            # Validation
            val=True,                 # Validate during training
            plots=True,               # Save plots
            
            # Resume training
            resume=resume,
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best model: runs/mask_helmet/train_v1/weights/best.pt")
        print(f"Last model: runs/mask_helmet/train_v1/weights/last.pt")
        print(f"Results: runs/mask_helmet/train_v1/")
        print("\nTraining metrics:")
        print(f"  Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print("\nNext steps:")
        print("  1. View results: start runs/mask_helmet/train_v1/results.png")
        print("  2. Test model: python test_mask_helmet.py")
        print("  3. Validate model: python -m ultralytics val model=runs/mask_helmet/train_v1/weights/best.pt")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("To resume: python train_mask_helmet.py --resume")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


def validate_trained_model(model_path, data_yaml):
    """Validate trained model on validation set"""
    print(f"Validating model: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    if hasattr(metrics.box, 'maps'):
        print("\nPer-class mAP50:")
        for i, map_val in enumerate(metrics.box.maps):
            print(f"  Class {i}: {map_val:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for mask/helmet detection')
    parser.add_argument('--data', type=str, default='data/mask_dataset.yaml',
                       help='Path to dataset YAML config')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (reduce if GPU memory error)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (0 for GPU, cpu for CPU, None for auto)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--validate', type=str, default=None,
                       help='Validate existing model (provide path to best.pt)')
    
    args = parser.parse_args()
    
    if args.validate:
        # Validation mode
        validate_trained_model(args.validate, args.data)
    else:
        # Training mode
        train_model(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            resume=args.resume
        )
