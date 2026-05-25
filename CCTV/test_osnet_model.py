"""
Test script for OSNet ImageNet pretrained model (osnet_x1_0_imagenet.pth)
Tests the model's ability to extract features from images
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
import time

# Try to import torchreid
try:
    import torchreid
    print("✓ torchreid imported successfully")
except ImportError:
    print("✗ torchreid not available. Install with: pip install torchreid")
    exit(1)


class OSNetTester:
    """Test OSNet ImageNet pretrained model"""
    
    def __init__(self, model_path='osnet_x1_0_imagenet.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model = None
        self.feature_dim = None
        
    def load_model(self):
        """Load the OSNet model from .pth file"""
        print(f"\n{'='*60}")
        print(f"Loading OSNet model from: {self.model_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(self.model_path):
            print(f"✗ Model file not found: {self.model_path}")
            return False
        
        try:
            # Load model architecture
            self.model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1000,
                pretrained=False,  # We'll load weights manually
                loss='softmax'
            )
            self.model.to(self.device)
            
            # Load weights from .pth file
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"✓ Model loaded successfully")
            
            # Get feature dimension
            self.feature_dim = 512  # OSNet outputs 512D features
            print(f"✓ Feature dimension: {self.feature_dim}D")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path, size=(256, 128)):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return None
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"✗ Failed to read image: {image_path}")
                return None
            
            # Resize to standard size
            img = cv2.resize(img, size)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize (ImageNet normalization)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            return img_tensor
            
        except Exception as e:
            print(f"✗ Error preprocessing image: {e}")
            return None
    
    def extract_features(self, image_path):
        """Extract features from image"""
        img_tensor = self.preprocess_image(image_path)
        if img_tensor is None:
            return None
        
        try:
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"✗ Error extracting features: {e}")
            return None
    
    def test_on_images(self, image_paths):
        """Test on multiple images and compute similarities"""
        print(f"\n{'='*60}")
        print(f"Testing on {len(image_paths)} image(s)")
        print(f"{'='*60}")
        
        features_list = []
        valid_paths = []
        
        # Extract features from all images
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {img_path}")
            
            features = self.extract_features(img_path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(img_path)
                print(f"✓ Features extracted: shape {features.shape}")
                print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
            else:
                print(f"✗ Failed to extract features")
        
        if len(features_list) < 2:
            print("\n⚠ Need at least 2 valid images to compute similarities")
            return
        
        # Compute pairwise similarities
        print(f"\n{'='*60}")
        print("Pairwise Cosine Similarities:")
        print(f"{'='*60}")
        
        for i in range(len(features_list)):
            for j in range(i+1, len(features_list)):
                feat_i = features_list[i].reshape(1, -1)
                feat_j = features_list[j].reshape(1, -1)
                
                # Cosine similarity
                similarity = np.dot(feat_i, feat_j.T)[0, 0]
                
                print(f"\n{os.path.basename(valid_paths[i])} ↔ {os.path.basename(valid_paths[j])}")
                print(f"  Cosine similarity: {similarity:.4f}")
                if similarity > 0.5:
                    print(f"  → High match (>0.5)")
                elif similarity > 0.3:
                    print(f"  → Medium match (0.3-0.5)")
                else:
                    print(f"  → Low match (<0.3)")
    
    def benchmark(self, num_iterations=10):
        """Benchmark model inference speed"""
        print(f"\n{'='*60}")
        print(f"Benchmarking (inference speed)")
        print(f"{'='*60}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000
        fps = 1000 / avg_time
        
        print(f"Average inference time: {avg_time:.2f}ms")
        print(f"Frames per second (single): {fps:.2f} fps")
        print(f"Batch processing (8 images): {fps/8:.2f} fps")


def find_test_images(max_images=5):
    """Find test images in common locations"""
    test_dirs = [
        'snapshots',
        'thumbnails',
        'thumbnails/suspects',
        '../test_images',
        'ingest'
    ]
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    found_images = []
    
    for test_dir in test_dirs:
        if os.path.isdir(test_dir):
            for file in os.listdir(test_dir):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    found_images.append(os.path.join(test_dir, file))
                    if len(found_images) >= max_images:
                        break
        if len(found_images) >= max_images:
            break
    
    return found_images[:max_images]


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     OSNet ImageNet Pretrained Model Test                  ║
║     Testing osnet_x1_0_imagenet.pth                        ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize tester
    tester = OSNetTester('osnet_x1_0_imagenet.pth')
    
    # Load model
    if not tester.load_model():
        exit(1)
    
    # Benchmark
    tester.benchmark(num_iterations=10)
    
    # Find test images
    print(f"\n{'='*60}")
    print("Looking for test images...")
    print(f"{'='*60}")
    
    test_images = find_test_images(max_images=5)
    
    if test_images:
        print(f"Found {len(test_images)} test image(s):")
        for img in test_images:
            print(f"  - {img}")
        tester.test_on_images(test_images)
    else:
        print("⚠ No test images found. Please provide image paths.")
        print("\nUsage example:")
        print("  python test_osnet_model.py --images image1.jpg image2.jpg")
        print("\nOr create a test image and place it in 'snapshots/' or 'thumbnails/' directory")
    
    print(f"\n{'='*60}")
    print("✓ Test completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
