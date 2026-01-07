"""
Small trainer wrapper for fine-tuning YOLOv8 on a mask dataset.

Usage examples:
  python train_mask.py --data data/mask_dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16

Notes:
- Dataset must be in YOLOv8 format: for each image.jpg there is image.txt with normalized bbox lines: <class> <x> <y> <w> <h>
- See `data/mask_dataset.yaml` for structure.
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/mask_dataset.yaml", help="dataset yaml path")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="base model or checkpoint")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--name", type=str, default="mask_finetune")
    p.add_argument("--device", type=str, default="cpu", help="cuda device index or cpu")
    p.add_argument("--workers", type=int, default=0, help="number of data loader workers (set 0 for low-memory/CPU)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Training YOLO from {args.model} on {args.data}")
    model = YOLO(args.model)

    # Enhanced training with better hyperparameters for mask/helmet detection
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        # Optimization
        patience=10,  # Early stopping
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
        # Better detection for small objects (masks on faces)
        conf=0.001,  # Low confidence during training to learn more
        iou=0.5,
        # Data augmentation (helps with limited data)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation
        hsv_v=0.4,    # Value
        degrees=10,   # Rotation
        translate=0.1,  # Translation
        scale=0.5,    # Scaling
        flipud=0.0,   # No vertical flip (faces are always upright)
        fliplr=0.5,   # Horizontal flip 50%
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
    )

    print("\n" + "="*60)
    print(f"Training complete! Best weights saved to:")
    print(f"  runs/detect/{args.name}/weights/best.pt")
    print("="*60)


if __name__ == "__main__":
    main()
