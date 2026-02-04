"""
Quick setup script for mask/helmet detection training
Creates necessary directories and verifies installation
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'albumentations': 'albumentations',
        'torch': 'torch',
        'yaml': 'pyyaml',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def create_directories():
    """Create necessary directory structure"""
    dirs = [
        'data/raw_images',
        'data/augmented_images',
        'data/labels',
        'data/mask_dataset/train/images',
        'data/mask_dataset/train/labels',
        'data/mask_dataset/val/images',
        'data/mask_dataset/val/labels',
        'data/mask_dataset/test/images',
        'data/mask_dataset/test/labels',
        'test_images',
        'test_results',
    ]
    
    print("\nCreating directories...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}")
    
    # Create classes.txt
    classes_file = 'data/labels/classes.txt'
    if not os.path.exists(classes_file):
        with open(classes_file, 'w') as f:
            f.write("mask\nhelmet\n")
        print(f"✓ Created {classes_file}")


def verify_dataset_yaml():
    """Verify dataset configuration file"""
    yaml_path = 'data/mask_dataset.yaml'
    
    if not os.path.exists(yaml_path):
        print(f"\n✗ Dataset config not found: {yaml_path}")
        print("Creating default configuration...")
        
        project_root = Path(__file__).parent.absolute()
        
        content = f"""# Dataset configuration for mask and helmet detection
# Update the 'path' if your dataset is stored elsewhere

# Absolute path to dataset directory
path: {str(project_root / 'data' / 'mask_dataset').replace(chr(92), '/')}

# Paths to train/val/test sets (relative to 'path')
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 2

# Class names (0-indexed)
names:
  0: mask
  1: helmet
"""
        with open(yaml_path, 'w') as f:
            f.write(content)
        print(f"✓ Created {yaml_path}")
    else:
        print(f"\n✓ Dataset config exists: {yaml_path}")


def show_next_steps():
    """Show next steps to user"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("\n1. COLLECT DATA:")
    print("   python scripts/extract_frames.py --video path/to/video.mp4")
    print("   (or download dataset from Kaggle/Roboflow)")
    
    print("\n2. AUGMENT DATA:")
    print("   python scripts/augment_dataset.py")
    
    print("\n3. LABEL DATA:")
    print("   pip install labelImg")
    print("   labelImg data/augmented_images data/labels")
    
    print("\n4. SPLIT DATASET:")
    print("   python scripts/split_dataset.py")
    
    print("\n5. TRAIN MODEL:")
    print("   python train_mask_helmet.py --model n --epochs 100")
    
    print("\n6. TEST MODEL:")
    print("   python test_mask_helmet.py --source test_images/sample.jpg")
    
    print("\nFor detailed instructions, see: TRAINING_GUIDE.md")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("MASK & HELMET DETECTION - SETUP")
    print("="*60)
    
    print("\n[1/4] Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing packages and run setup again.")
        sys.exit(1)
    
    print("\n[2/4] Creating directory structure...")
    create_directories()
    
    print("\n[3/4] Verifying dataset configuration...")
    verify_dataset_yaml()
    
    print("\n[4/4] Setup validation...")
    show_next_steps()
