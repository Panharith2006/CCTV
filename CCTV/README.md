# CCTV AI - Intelligent Video Surveillance System


#### Layer 1: Frame Ingestion
- ✅ Multi-camera video stream processing
- ✅ RTSP stream support
- ✅ Frame extraction and preprocessing
- ✅ Configurable frame rate control

#### Layer 2: Detection System
- ✅ YOLOv8-based person detection
- ✅ Two-stage detection pipeline (YOLO + ReID)
- ✅ Real-time bounding box extraction
- ✅ Feature extraction for Re-ID

#### Layer 3: Tracking System
- ✅ SORT (Simple Online and Realtime Tracking) implementation
- ✅ Multi-object tracking across frames
- ✅ Re-ID manager for person matching across cameras
- ✅ Identity persistence and recovery

#### Layer 4: Motion Analysis
- ✅ Motion pattern tracking
- ✅ Person trajectory analysis
- ✅ Cross-camera movement detection

#### Layer 5: Behavior Analysis
- ✅ Behavioral pattern recognition
- ✅ Activity classification
- ✅ Event detection system

#### Layer 6: Alert System
- ✅ Telegram bot integration
- ✅ Real-time alert notifications
- ✅ Configurable alert triggers

#### Database Integration
- ✅ MySQL database for person storage
- ✅ Re-ID feature vector storage
- ✅ Person enrollment system
- ✅ Historical data tracking
- ✅ Mask and helmet detection status storage

#### Safety Equipment Monitoring
- ✅ Database schema for mask/helmet detection
- ⏳ Mask detection model integration (In Progress)
- ⏳ Helmet detection model integration (In Progress)

### Current Capabilities
- Detect and track multiple persons across multiple cameras
- Maintain identity consistency using Re-ID features
- Store and retrieve person information from MySQL database
- Send alerts via Telegram for specific events
- Track person movements and behaviors
- Database ready for mask and helmet detection data



## 🔮 Next Phase Development

### Phase 1: Enhanced Safety Detection (Current Priority)
**Timeline: 2-4 weeks**

#### 1.1 Mask Detection Model Fine-Tuning
- Train custom YOLOv8 model for face mask detection
- Achieve >95% accuracy on mask/no-mask classification
- Integrate with existing Layer 2 detection pipeline
- Real-time mask compliance monitoring

#### 1.2 Helmet Detection Model Fine-Tuning
- Train custom YOLOv8 model for safety helmet detection
- Support multiple helmet types (construction, motorcycle, industrial)
- Integrate with existing Layer 2 detection pipeline
- Zone-based helmet requirement enforcement

#### 1.3 Safety Compliance Dashboard
- Real-time safety equipment compliance statistics
- Per-person safety history tracking
- Zone-based safety rule configuration
- Automated violation reporting

### Phase 2: Web-Based Frontend Interface
**Timeline: 4-6 weeks**

#### 2.1 Frontend Technology Stack
- **Framework**: React.js with TypeScript
- **UI Library**: Material-UI (MUI) or Ant Design
- **Real-time Communication**: WebSocket/Socket.io
- **Video Streaming**: WebRTC or HLS.js
- **State Management**: Redux Toolkit or Zustand
- **Data Visualization**: Chart.js or Recharts

#### 2.2 Core Frontend Features

##### Live Monitoring Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  CCTV AI - Live Monitoring Dashboard                        │
├─────────────────────────────────────────────────────────────┤
│  🎥 Camera Grid View (2x2, 3x3, 4x4 layouts)               │
│  ├── Real-time video streams with detection overlays        │
│  ├── Bounding boxes for detected persons                    │
│  ├── Person ID labels and Re-ID status                      │
│  ├── Mask/Helmet status indicators (✓ or ✗)               │
│  └── Click to focus on specific camera                      │
├─────────────────────────────────────────────────────────────┤
│  📊 Real-time Statistics Panel                              │
│  ├── Total persons detected (per camera & overall)          │
│  ├── Active tracks count                                    │
│  ├── Mask compliance rate (%)                               │
│  ├── Helmet compliance rate (%)                             │
│  └── Alert status and count                                 │
└─────────────────────────────────────────────────────────────┘
```

##### Person Management Interface
- Search and filter persons by ID, timestamp, camera
- View person's movement history across cameras
- Display person thumbnails and Re-ID feature similarity
- Safety equipment compliance history per person
- Manual person enrollment interface
- Edit/update person metadata

##### Alert & Event Management
- Real-time alert feed with filtering
- Alert priority levels and categories
- Acknowledge and dismiss alerts
- Alert history and analytics
- Configure alert rules and triggers
- Telegram notification settings

##### Analytics & Reports
- Daily/weekly/monthly safety compliance reports
- Person traffic heatmaps
- Camera performance metrics
- Detection accuracy statistics
- Zone-based analytics
- Export reports as PDF/CSV

##### System Configuration
- Camera management (add/remove/edit cameras)
- Detection model settings (confidence thresholds)
- Re-ID sensitivity configuration
- Database connection settings
- User management and permissions
- System health monitoring

#### 2.3 Backend API Development
- RESTful API with FastAPI or Flask
- WebSocket server for real-time updates
- Video streaming server (RTSP to WebRTC bridge)
- Authentication and authorization (JWT)
- Database query optimization
- Caching layer (Redis) for performance

#### 2.4 Deployment Architecture
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│   React Client   │────▶│   Nginx Proxy    │────▶│  FastAPI     │
│   (Browser)      │◀────│   (Port 80/443)  │◀────│  Backend     │
└──────────────────┘     └──────────────────┘     └──────┬───────┘
                                                           │
         ┌─────────────────────────────────────────────────┤
         │                                                  │
    ┌────▼────┐                                       ┌────▼────┐
    │  MySQL  │                                       │  CCTV   │
    │Database │                                       │ AI Core │
    └─────────┘                                       └─────────┘
```

### Phase 3: Advanced Features (Future)
**Timeline: 8-12 weeks**

- Edge computing support (run detection on edge devices)
- Multi-site deployment and management
- Advanced behavior analysis (loitering, crowd detection)
- Integration with access control systems
- Mobile application (iOS/Android)
- AI model performance monitoring
- Automated model retraining pipeline
- Privacy compliance features (face blurring, GDPR)

---

## 🎯 Fine-Tuning for Mask & Helmet Detection

This section provides a comprehensive beginner-friendly guide for training custom YOLOv8 models for mask and helmet detection.

### Prerequisites

#### Required Software
```bash
# Python 3.8+ (already installed)
# Ultralytics YOLOv8
pip install ultralytics

# Additional tools for data preparation
pip install labelImg  # For manual annotation
pip install roboflow  # For dataset management (optional)
```

#### Hardware Requirements
- **Minimum**: CPU with 8GB RAM (slow training)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **Optimal**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100)

### Step 1: Understanding the Data Requirements

#### Dataset Structure
Your dataset should follow this structure:
```
data/
├── mask_helmet_dataset/
│   ├── train/
│   │   ├── images/
│   │   │   ├── img_001.jpg
│   │   │   ├── img_002.jpg
│   │   │   └── ...
│   │   └── labels/
│   │       ├── img_001.txt
│   │       ├── img_002.txt
│   │       └── ...
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

#### Label Format (YOLO Format)
Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1 range):
- `class_id`: Integer class index (e.g., 0=no_mask, 1=mask, 2=no_helmet, 3=helmet)
- `x_center, y_center`: Center point of bounding box (relative to image width/height)
- `width, height`: Bounding box dimensions (relative to image width/height)

**Example label file (img_001.txt):**
```
1 0.5 0.3 0.15 0.20    # Person with mask at center
3 0.7 0.4 0.12 0.18    # Person with helmet on right
```

### Step 2: Data Collection Strategy

#### Option 1: Use Existing Datasets
Download pre-annotated datasets:
- **Face Mask Detection**: Kaggle, Roboflow Universe
- **Helmet Detection**: Roboflow, Open Images Dataset

```python
# Example: Download from Roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("mask-detection")
dataset = project.version(1).download("yolov8")
```

#### Option 2: Collect Custom Data
1. **Record CCTV Footage**: Extract frames from your actual cameras
2. **Extract Frames**: Use FFmpeg or OpenCV
3. **Ensure Diversity**:
   - Different lighting conditions (day/night/indoor/outdoor)
   - Various angles and distances
   - Different mask/helmet types and colors
   - Multiple persons per frame
   - Occlusions and partial views

```python
# Extract frames from video for annotation
import cv2

video_path = "cctv_footage.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save every 30th frame (1 frame per second for 30fps video)
    if frame_count % 30 == 0:
        cv2.imwrite(f"data/raw_images/frame_{frame_count:06d}.jpg", frame)
    
    frame_count += 1

cap.release()
```

### Step 3: Data Annotation

#### Using LabelImg (Recommended for Beginners)
```bash
# Install LabelImg
pip install labelImg

# Run LabelImg
labelImg
```

**LabelImg Workflow:**
1. Open `data/raw_images` directory
2. Set save directory to appropriate `labels/` folder
3. Click "Create RectBox" to draw bounding boxes
4. Select class (mask, no_mask, helmet, no_helmet)
5. Press Ctrl+S to save
6. Press 'D' to go to next image

#### Annotation Best Practices
- **Tight Bounding Boxes**: Box should closely fit the mask/helmet
- **Consistency**: Use same criteria across all images
- **Edge Cases**: Include partially visible objects
- **Multiple Objects**: Annotate all visible masks/helmets in frame
- **Quality Control**: Review annotations after every 50-100 images

### Step 4: Dataset Preparation

#### Create Dataset Configuration File
Create `data/mask_helmet_dataset.yaml`:

```yaml
# Dataset configuration for mask and helmet detection

# Paths (relative to this file or absolute)
path: D:/AHBD/CCTV_AI/data/mask_helmet_dataset
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: no_mask
  1: with_mask
  2: no_helmet
  3: with_helmet

# Number of classes
nc: 4
```

#### Split Dataset (Train/Validation/Test)
Recommended split:
- **Training**: 70-80% (used for learning)
- **Validation**: 10-15% (used during training to tune)
- **Test**: 10-15% (used after training to evaluate)

```python
# Automated dataset splitting script
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_images, source_labels, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_images: Path to images folder
        source_labels: Path to labels folder
        output_dir: Output directory for split dataset
        split_ratio: Tuple of (train, val, test) ratios
    """
    # Create output directories
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
    
    # Get all image files
    image_files = list(Path(source_images).glob("*.jpg")) + \
                  list(Path(source_images).glob("*.png"))
    
    # Shuffle
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    # Split files
    splits_data = {
        'train': image_files[:train_end],
        'valid': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Copy files
    for split_name, files in splits_data.items():
        for img_path in files:
            # Copy image
            shutil.copy(img_path, f"{output_dir}/{split_name}/images/")
            
            # Copy corresponding label
            label_path = Path(source_labels) / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, f"{output_dir}/{split_name}/labels/")
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(splits_data['train'])} images")
    print(f"  Valid: {len(splits_data['valid'])} images")
    print(f"  Test: {len(splits_data['test'])} images")

# Usage
split_dataset(
    source_images="data/raw_images",
    source_labels="data/raw_labels",
    output_dir="data/mask_helmet_dataset"
)
```

### Step 5: Fine-Tuning the Model

#### Training Script
Create `train_mask_helmet.py`:

```python
from ultralytics import YOLO
import torch

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# Load a pretrained YOLOv8 model
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
# Start with 'n' (nano) for fast training, use 's' or 'm' for better accuracy
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data/mask_helmet_dataset.yaml',  # Path to dataset config
    epochs=100,                             # Number of training epochs
    imgsz=640,                              # Input image size
    batch=16,                               # Batch size (reduce if GPU memory error)
    patience=20,                            # Early stopping patience
    save=True,                              # Save checkpoints
    device=device,                          # Training device
    workers=4,                              # Number of data loading workers
    project='runs/mask_helmet',             # Project directory
    name='train_v1',                        # Experiment name
    exist_ok=False,                         # Don't overwrite existing
    pretrained=True,                        # Use pretrained weights
    optimizer='AdamW',                      # Optimizer (AdamW, SGD, Adam)
    lr0=0.01,                               # Initial learning rate
    weight_decay=0.0005,                    # Weight decay
    warmup_epochs=3,                        # Warmup epochs
    cos_lr=True,                            # Use cosine learning rate scheduler
    mosaic=1.0,                             # Mosaic augmentation probability
    mixup=0.1,                              # Mixup augmentation probability
    copy_paste=0.0,                         # Copy-paste augmentation
    degrees=10.0,                           # Image rotation (+/- deg)
    translate=0.1,                          # Image translation (+/- fraction)
    scale=0.5,                              # Image scale (+/- gain)
    fliplr=0.5,                             # Horizontal flip probability
    hsv_h=0.015,                            # Hue augmentation
    hsv_s=0.7,                              # Saturation augmentation
    hsv_v=0.4,                              # Value augmentation
)

# Validate the model
metrics = model.val()

print("\n=== Training Complete ===")
print(f"Best model saved to: runs/mask_helmet/train_v1/weights/best.pt")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

#### Running Training
```bash
# Activate your Python environment
# Run training script
python train_mask_helmet.py
```

#### Training Hyperparameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `epochs` | Number of complete passes through dataset | 50-300 |
| `imgsz` | Input image size (square) | 640 (default), 416, 1280 |
| `batch` | Images processed together | 8-64 (depends on GPU) |
| `lr0` | Initial learning rate | 0.001-0.01 |
| `patience` | Epochs to wait before early stopping | 20-50 |
| `mosaic` | Mosaic augmentation (combine 4 images) | 0.0-1.0 |
| `degrees` | Rotation augmentation range | 0-20 degrees |
| `fliplr` | Horizontal flip probability | 0.0-0.5 |

### Step 6: Evaluating the Model

#### Validation Metrics
After training, check these metrics in `runs/mask_helmet/train_v1/results.png`:

- **mAP50**: Mean Average Precision at IoU=0.50 (target: >0.90)
- **mAP50-95**: mAP averaged over IoU 0.50-0.95 (target: >0.75)
- **Precision**: Correct detections / All detections (target: >0.85)
- **Recall**: Correct detections / All ground truths (target: >0.85)
- **Loss**: Training and validation loss (should decrease)

#### Test on New Images
```python
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('runs/mask_helmet/train_v1/weights/best.pt')

# Test on single image
results = model.predict('test_image.jpg', conf=0.5)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        print(f"Detected: {class_name} (confidence: {confidence:.2f})")

# Save annotated image
annotated = results[0].plot()
cv2.imwrite('result.jpg', annotated)
```

#### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Low mAP (<0.60) | Insufficient data | Collect more images (aim for 1000+) |
| Overfitting | Model memorizing training data | Increase data augmentation, use larger dataset |
| GPU Out of Memory | Batch size too large | Reduce `batch` parameter (try 8 or 4) |
| Slow training | CPU training | Use GPU or reduce image size |
| Poor detection on small objects | Low resolution | Increase `imgsz` to 1280, collect more close-up data |

### Step 7: Integration with CCTV System

#### Update Layer 2 Detector
Modify `detector/layer2_yolo_detector.py` to include mask/helmet detection:

```python
from ultralytics import YOLO

class EnhancedYOLODetector:
    def __init__(self):
        # Person detection model
        self.person_model = YOLO('yolov8n.pt')
        
        # Mask and helmet detection model
        self.safety_model = YOLO('runs/mask_helmet/train_v1/weights/best.pt')
    
    def detect(self, frame):
        # First, detect persons
        person_results = self.person_model.predict(frame, classes=[0])  # class 0 = person
        
        # For each detected person, check for mask/helmet
        safety_detections = []
        for person_box in person_results[0].boxes:
            x1, y1, x2, y2 = map(int, person_box.xyxy[0])
            
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            
            # Detect mask/helmet in person region
            safety_results = self.safety_model.predict(person_crop, conf=0.5)
            
            # Extract safety equipment status
            has_mask = any(int(box.cls[0]) == 1 for box in safety_results[0].boxes)
            has_helmet = any(int(box.cls[0]) == 3 for box in safety_results[0].boxes)
            
            safety_detections.append({
                'person_box': (x1, y1, x2, y2),
                'has_mask': has_mask,
                'has_helmet': has_helmet
            })
        
        return safety_detections
```

#### Update Database Storage
When storing person detections, include mask/helmet status (already supported in your database schema).

### Step 8: Continuous Improvement

#### Model Iteration Workflow
1. **Collect Edge Cases**: Save frames where model performs poorly
2. **Annotate New Data**: Add to training set
3. **Retrain**: Run training with expanded dataset
4. **Compare Metrics**: Ensure improvement over previous version
5. **Deploy**: Replace model in production if better
6. **Monitor**: Track real-world performance

#### Data Quality Checklist
✅ At least 500-1000 images per class  
✅ Balanced class distribution (similar number of each class)  
✅ Diverse lighting and weather conditions  
✅ Various camera angles and distances  
✅ Real CCTV footage (not stock photos)  
✅ Consistent annotation quality  
✅ No duplicate or near-duplicate images  
✅ Proper train/val/test split  

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/CCTV_AI.git
cd CCTV_AI
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure MySQL Database
Edit `config/mysql_config.py`:
```python
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'cctv_ai'
}
```

Run migration to add mask/helmet columns:
```bash
python scripts/migrate_add_mask_helmet_columns.py
```

### 5. Configure Telegram Bot
Edit `config/telegram_config.py`:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

### 6. Configure Cameras
Edit `config/layer0_cameras.py` with your camera RTSP URLs:
```python
CAMERAS = [
    {
        'id': 1,
        'name': 'Entrance Camera',
        'url': 'rtsp://username:password@camera_ip:554/stream'
    },
    # Add more cameras...
]
```

---

## 🚀 Usage

### Running the Main System
```bash
# Run main detection and tracking system
python main.py
```

### Debug Mode
```bash
# Run with visualization for debugging
python debug_infer.py
```

### Training Models
```bash
# Train mask and helmet detection model
python train_mask_helmet.py

# Validate trained model
python train_validate.py
```

### Testing Individual Layers
```bash
# Test frame ingestion
python test/test_layer1.py

# Test YOLO detection
python test/test_layer2.py

# Test tracking
python test/test_layer3.py

# Test motion tracking
python test/test_layer4.py

# Test Re-ID system
python test/test_reid.py
```

### Database Management
```bash
# View all enrolled persons
python scripts/print_db.py

# Enroll new person
python tools/enroll_person.py

# Review stored persons
python tools/review_persons.py
```

### System Health Check
```bash
python test/test_system_health.py
```

---

## 📁 Project Structure

```
CCTV_AI/
├── config/                      # Configuration files
│   ├── layer0_cameras.py       # Camera RTSP URLs
│   ├── mysql_config.py         # Database configuration
│   └── telegram_config.py      # Telegram bot settings
│
├── database/                    # Database modules
│   ├── reid_database.py        # File-based database
│   └── reid_database_mysql.py  # MySQL database integration
│
├── detector/                    # Detection layer
│   ├── layer2_yolo_detector.py # YOLOv8 person detection
│   ├── layer2_reid_extractor.py # ReID feature extraction
│   └── two_stage_detector.py   # Combined detection pipeline
│
├── ingest/                      # Frame ingestion
│   └── layer1_frame_ingest.py  # Video stream processing
│
├── tracker/                     # Tracking and analysis
│   ├── layer3_sort_tracker.py  # SORT tracker implementation
│   ├── layer3_reid_manager.py  # Re-ID management
│   ├── layer4_motion_tracker.py # Motion analysis
│   ├── layer5_behavior.py      # Behavior analysis
│   └── layer6_telegram.py      # Alert notifications
│
├── sort/                        # SORT algorithm
│   └── sort.py                 # SORT tracking implementation
│
├── tools/                       # Utility tools
│   ├── enroll_person.py        # Person enrollment
│   └── review_persons.py       # Person review interface
│
├── scripts/                     # Database migration scripts
│   ├── migrate_add_mask_helmet_columns.py
│   └── print_db.py
│
├── test/                        # Test scripts
│   ├── test_layer1.py          # Test frame ingestion
│   ├── test_layer2.py          # Test detection
│   ├── test_layer3.py          # Test tracking
│   ├── test_layer4.py          # Test motion analysis
│   ├── test_reid.py            # Test Re-ID
│   └── test_system_health.py   # System health check
│
├── data/                        # Dataset storage
│   ├── mask_dataset.yaml       # Dataset configuration
│   └── mask_helmet_dataset/    # Training data
│
├── thumbnails/                  # Person thumbnails
│
├── main.py                      # Main application entry
├── debug_infer.py              # Debug visualization
├── train_mask_helmet.py        # Model training (to be created)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

[Add your license here]

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- SORT tracking algorithm
- Re-ID models and research community
- Open source contributors

---

**Last Updated**: January 31, 2026  
**Version**: 1.0.0
