"""
Augment dataset images for better model generalization
Applies transformations: rotation, brightness, blur, noise, etc.
Usage: python scripts/augment_dataset.py --input data/raw_images --output data/augmented_images
"""

import albumentations as A
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def create_augmentation_pipeline():
    """Create augmentation pipeline for mask/helmet detection"""
    return A.Compose([
        # Geometric transformations
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        
        # Color/lighting variations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Blur and noise (simulate camera quality issues)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Weather/environment effects
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.2),
        
        # Quality degradation
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
    ])


def augment_dataset(input_dir, output_dir, augmentations_per_image=3, keep_original=True):
    """
    Augment all images in input directory
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        augmentations_per_image: Number of augmented versions per original
        keep_original: If True, also copy original to output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Generating {augmentations_per_image} augmentations per image")
    print(f"Total output: {len(image_files) * (augmentations_per_image + (1 if keep_original else 0))} images\n")
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Process each image
    for img_path in tqdm(image_files, desc="Augmenting"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Save original (optional)
        if keep_original:
            output_path = os.path.join(output_dir, f"{img_path.stem}_orig{img_path.suffix}")
            cv2.imwrite(output_path, image)
        
        # Generate augmented versions
        for i in range(augmentations_per_image):
            try:
                augmented = transform(image=image)['image']
                output_path = os.path.join(output_dir, f"{img_path.stem}_aug{i:02d}{img_path.suffix}")
                cv2.imwrite(output_path, augmented)
            except Exception as e:
                print(f"Warning: Failed to augment {img_path.name}: {e}")
    
    print(f"\nAugmentation complete!")
    print(f"Output saved to: {output_dir}")
    
    # Count final images
    output_files = list(Path(output_dir).glob("*.*"))
    print(f"Total images in output: {len(output_files)}")


def preview_augmentations(image_path, num_samples=6):
    """
    Preview augmentation effects on a single image
    Useful for tuning augmentation parameters
    """
    import matplotlib.pyplot as plt
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = create_augmentation_pipeline()
    
    # Create subplot grid
    rows = 2
    cols = (num_samples + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, num_samples):
        augmented = transform(image=image)['image']
        augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        axes[i].imshow(augmented_rgb)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Preview saved to: augmentation_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment dataset images')
    parser.add_argument('--input', type=str, default='data/raw_images',
                       help='Input directory with original images')
    parser.add_argument('--output', type=str, default='data/augmented_images',
                       help='Output directory for augmented images')
    parser.add_argument('--num-aug', type=int, default=3,
                       help='Number of augmentations per image (default: 3)')
    parser.add_argument('--keep-original', action='store_true', default=True,
                       help='Keep original images in output')
    parser.add_argument('--preview', type=str, default=None,
                       help='Preview augmentations on single image')
    
    args = parser.parse_args()
    
    if args.preview:
        # Preview mode
        preview_augmentations(args.preview)
    else:
        # Augment entire dataset
        augment_dataset(
            input_dir=args.input,
            output_dir=args.output,
            augmentations_per_image=args.num_aug,
            keep_original=args.keep_original
        )
