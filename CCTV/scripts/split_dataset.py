"""
Split labeled dataset into train/val/test sets
Ensures corresponding images and labels stay together
Usage: python scripts/split_dataset.py --images data/augmented_images --labels data/labels --output data/mask_dataset
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels (.txt)
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
    
    print(f"Splitting dataset:")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Val:   {val_ratio*100:.1f}%")
    print(f"  Test:  {test_ratio*100:.1f}%\n")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Check for corresponding labels
    images_with_labels = []
    images_without_labels = []
    
    for img_path in image_files:
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            images_with_labels.append(img_path)
        else:
            images_without_labels.append(img_path)
    
    print(f"Images with labels: {len(images_with_labels)}")
    print(f"Images without labels: {len(images_without_labels)} (will be skipped)")
    
    if len(images_with_labels) == 0:
        print(f"Error: No labeled images found. Check labels directory: {labels_dir}")
        return
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(images_with_labels)
    
    # Calculate split indices
    total = len(images_with_labels)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split files
    splits_data = {
        'train': images_with_labels[:train_end],
        'val': images_with_labels[train_end:val_end],
        'test': images_with_labels[val_end:]
    }
    
    # Copy files
    print("\nCopying files...")
    stats = defaultdict(lambda: {'images': 0, 'labels': 0, 'classes': defaultdict(int)})
    
    for split_name, files in splits_data.items():
        print(f"Processing {split_name} set...")
        
        for img_path in files:
            # Copy image
            img_dest = f"{output_dir}/{split_name}/images/{img_path.name}"
            shutil.copy(img_path, img_dest)
            stats[split_name]['images'] += 1
            
            # Copy label
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            label_dest = f"{output_dir}/{split_name}/labels/{label_path.name}"
            shutil.copy(label_path, label_dest)
            stats[split_name]['labels'] += 1
            
            # Count classes in label
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            class_id = int(parts[0])
                            stats[split_name]['classes'][class_id] += 1
            except Exception as e:
                print(f"Warning: Could not parse label {label_path.name}: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("SPLIT COMPLETE!")
    print("="*60)
    
    for split_name in splits:
        s = stats[split_name]
        print(f"\n{split_name.upper()} SET:")
        print(f"  Images: {s['images']}")
        print(f"  Labels: {s['labels']}")
        if s['classes']:
            print(f"  Class distribution:")
            for class_id, count in sorted(s['classes'].items()):
                print(f"    Class {class_id}: {count} instances")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Verify the split looks correct")
    print("  2. Update data/mask_dataset.yaml with correct paths")
    print("  3. Run training: python train_mask_helmet.py")


def analyze_dataset(dataset_dir):
    """
    Analyze dataset statistics (useful for checking split quality)
    """
    print(f"\nAnalyzing dataset: {dataset_dir}\n")
    
    splits = ['train', 'val', 'test']
    class_names = {0: 'mask', 1: 'helmet'}  # Update based on your classes
    
    for split in splits:
        labels_dir = f"{dataset_dir}/{split}/labels"
        if not os.path.exists(labels_dir):
            continue
        
        label_files = list(Path(labels_dir).glob("*.txt"))
        total_instances = 0
        class_counts = defaultdict(int)
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_instances += 1
        
        print(f"{split.upper()} SET:")
        print(f"  Files: {len(label_files)}")
        print(f"  Total instances: {total_instances}")
        for class_id, count in sorted(class_counts.items()):
            class_name = class_names.get(class_id, f"Class_{class_id}")
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--images', type=str, default='data/augmented_images',
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, default='data/labels',
                       help='Directory containing YOLO format labels')
    parser.add_argument('--output', type=str, default='data/mask_dataset',
                       help='Output directory for split dataset')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing dataset instead of splitting')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.output)
    else:
        split_dataset(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )
