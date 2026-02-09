# Mask & Helmet Detection Training Guide

This guide will walk you through training a custom YOLOv8 model for mask and helmet detection in your CCTV AI system.

## Quick Start

### 1. Data Collection (Choose one option)

#### Option A: Extract from existing CCTV footage
```powershell
# Extract frames from video
python scripts/extract_frames.py --video path/to/footage.mp4 --output data/raw_images

# Or from live camera
python scripts/extract_frames.py --camera 0 --duration 120 --output data/raw_images
```

#### Option B: Download public dataset
```powershell
# Install roboflow
pip install roboflow

# Download dataset (get API key from roboflow.com)
python -c "from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); project = rf.workspace().project('mask-detection'); dataset = project.version(1).download('yolov8', location='data/raw_dataset')"
```

### 2. Data Augmentation
```powershell
# Augment images (creates 3x more training data)
python scripts/augment_dataset.py --input data/raw_images --output data/augmented_images --num-aug 3

# Preview augmentations on single image (optional)
python scripts/augment_dataset.py --preview data/raw_images/sample.jpg
```

### 3. Label Images

#### Install LabelImg
```powershell
pip install labelImg
```

#### Start Labeling
```powershell
# Launch LabelImg
labelImg data/augmented_images data/labels
```

**LabelImg Instructions:**
1. Click "Open Dir" → Select `data/augmented_images`
2. Click "Change Save Dir" → Select `data/labels`
3. Switch format to "YOLO" (from PascalVOC)
4. Create `data/labels/classes.txt`:
   ```
   mask
   helmet
   ```
5. **Label each image:**
   - Press `W` to draw bounding box
   - Draw box around head/face when mask visible
   - Draw box around head when helmet visible
   - Select correct class
   - Press `Ctrl+S` to save
   - Press `D` for next image

**Labeling Tips:**
- Tight boxes around head region
- Label all visible masks/helmets in frame
- Include partially visible objects (>30%)
- Maintain consistency across all images

### 4. Split Dataset
```powershell
# Split into train (70%), val (20%), test (10%)
python scripts/split_dataset.py --images data/augmented_images --labels data/labels --output data/mask_dataset

# Analyze the split (optional)
python scripts/split_dataset.py --output data/mask_dataset --analyze
```

### 5. Train Model
```powershell
# Train YOLOv8 nano (fastest)
python train_mask_helmet.py --model n --epochs 100 --batch 16

# Or train YOLOv8 small (better accuracy)
python train_mask_helmet.py --model s --epochs 150 --batch 16

# Resume training if interrupted
python train_mask_helmet.py --resume
```

**Training Parameters:**
- `--model n`: Nano (fastest, 3ms inference)
- `--model s`: Small (balanced, 4ms inference)
- `--model m`: Medium (accurate, 8ms inference)
- `--batch 16`: Reduce to 8 or 4 if GPU memory error
- `--epochs 100`: Increase to 200 for better results

### 6. Monitor Training
```powershell
# View training curves
start runs/mask_helmet/train_v1/results.png

# View confusion matrix
start runs/mask_helmet/train_v1/confusion_matrix.png

# View validation predictions
start runs/mask_helmet/train_v1/val_batch0_pred.jpg
```

### 7. Test Model
```powershell
# Test on single image
python test_mask_helmet.py --source test_images/sample.jpg

# Test on video
python test_mask_helmet.py --source test_video.mp4

# Test on directory of images
python test_mask_helmet.py --source test_images/

# Test on live camera
python test_mask_helmet.py --source 0

# Test on RTSP stream
python test_mask_helmet.py --source "rtsp://admin:pass@192.168.1.100:554/stream"
```

## File Structure After Setup

```
CCTV_AI/
├── data/
│   ├── raw_images/              # Original extracted frames
│   ├── augmented_images/        # Augmented images
│   ├── labels/                  # YOLO format labels
│   └── mask_dataset/           # Final split dataset
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── runs/
│   └── mask_helmet/
│       └── train_v1/
│           ├── weights/
│           │   ├── best.pt     # Best model checkpoint
│           │   └── last.pt     # Last epoch checkpoint
│           ├── results.png     # Training curves
│           └── confusion_matrix.png
├── test_results/               # Test output images/videos
├── scripts/
│   ├── extract_frames.py
│   ├── augment_dataset.py
│   └── split_dataset.py
├── train_mask_helmet.py
└── test_mask_helmet.py
```

## Training Metrics to Watch

**Good Model Indicators:**
- `mAP50` > 0.90 (90% mean Average Precision)
- `Precision` > 0.85 (few false positives)
- `Recall` > 0.85 (few missed detections)
- `box_loss` < 0.5 (good bounding box accuracy)
- `cls_loss` < 0.3 (good classification)

**If metrics are poor:**
1. Collect more diverse training data (different lighting, angles, distances)
2. Increase epochs to 200+
3. Verify label quality
4. Ensure balanced dataset (equal mask/helmet samples)
5. Try larger model (yolov8s or yolov8m)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **GPU Out of Memory** | Reduce `--batch` to 8 or 4 |
| **Training too slow** | Use Google Colab (free GPU) or reduce image size |
| **Low accuracy (<80%)** | Need more diverse data, increase epochs |
| **High loss, not decreasing** | Check labels are correct, reduce learning rate |
| **Overfitting** | Add more augmentation, collect more data |
| **Model not detecting** | Lower confidence threshold in test |

## Integration with CCTV AI System

Once trained, integrate the model into your system:

1. **Update `detector/two_stage_detector.py`:**
   ```python
   # Use your trained model
   attr_model = YOLO('runs/mask_helmet/train_v1/weights/best.pt')
   ```

2. **Test integration:**
   ```powershell
   python debug_infer.py
   ```

3. **Update confidence thresholds** in `two_stage_detector.py` if needed:
   ```python
   conf_attr=0.3  # Adjust based on your model performance
   ```

## Next Steps After Training

1. ✅ Validate model performance meets requirements
2. ✅ Integrate into CCTV AI pipeline
3. ✅ Test on real CCTV streams
4. ✅ Add temporal smoothing (reduce false positives)
5. ✅ Implement storage logic (store only if mask AND helmet)
6. ✅ Add alert system for safety compliance

---

**Need Help?** Check `TROUBLESHOOTING.md` or raise an issue.
