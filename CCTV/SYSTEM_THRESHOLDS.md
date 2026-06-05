# System Thresholds Configuration

This document lists all thresholds used in the CCTV AI Security System.

---

## 1. DETECTION THRESHOLDS (YOLO Model)

### Person Detection
- **Threshold**: `0.40` (40% confidence) — **LOWERED for better security**
- **Location**: `detector/single_stage_detector.py` → `conf_person`
- **Purpose**: Confidence threshold for detecting persons in the frame
- **Effect**: 
  - Higher → Fewer false positives, may miss some people ⚠️
  - Lower → More detections, catches more people ✓ (Better for security)
- **Why 0.40?**: Security systems should err on detecting MORE people rather than missing someone

### Mask Detection
- **Threshold**: `0.50` (50% confidence)
- **Location**: `detector/single_stage_detector.py` → `conf_mask`
- **Purpose**: Confidence threshold for detecting masks on head crops
- **Effect**:
  - Higher → Fewer false mask detections, may miss some masks
  - Lower → More mask detections, possible false positives
- **Why 0.50?**: Balanced threshold since mask = WARNING (less critical than helmet)

### Helmet Detection
- **Threshold**: `0.65` (65% confidence) — **HIGHER to filter model inaccuracy**
- **Location**: `detector/single_stage_detector.py` → `conf_helmet`
- **Purpose**: Confidence threshold for detecting helmets on head crops
- **Effect**:
  - CRITICAL: Model may falsely detect hair/hoods/caps as helmets (false positives)
  - Lower threshold = accepts weak false detections = person WITHOUT helmet seen as "has helmet" = MISSED violations
  - Higher threshold = only accepts confident detections = filters false positives = accurate violation detection
- **Why 0.65?**: Model accuracy issues require strict filtering
  - Prevents false helmet detections that would mask real violations
  - Better to occasionally alert on actual helmet (false alarm) than miss violations due to false detections

### Head Region Crop
- **Threshold**: `0.45` (top 45% of person bounding box)
- **Location**: `detector/single_stage_detector.py` → `head_fraction`
- **Purpose**: Defines head region for targeted mask/helmet detection
- **Effect**:
  - Higher (e.g., 0.5) → Larger head region, may include shoulders
  - Lower (e.g., 0.4) → Smaller head region, may miss some headwear

---

## 2. RE-IDENTIFICATION (ReID) THRESHOLDS

### Feature Matching Similarity
- **Threshold**: `0.62` (62% similarity) — **UPDATED for better accuracy**
- **Location**: `tracker/violation_only_reid.py` → `similarity_threshold`
- **Purpose**: Cosine similarity threshold for matching person feature vectors
- **Effect**:
  - Higher (e.g., 0.7) → More strict matching, less re-identification (more unique IDs)
  - Lower (e.g., 0.5) → More lenient matching, more re-identification (fewer unique IDs)
- **Note**: Using cosine similarity, range is -1 to 1 (typically 0.3-0.9 for person matching)
- **Why 0.62?**: For violation-only system, false matches are worse than missed re-IDs
  - False match = Innocent person marked as violator ❌
  - Missed re-ID = Same violator gets 2 IDs (acceptable) ✓

### Feature Quality
- **Threshold**: `0.25` (25% quality score)
- **Location**: `detector/layer2_reid_extractor_enhanced.py` → `quality_threshold`
- **Purpose**: Minimum quality for extracted ReID features (blur detection, size check)
- **Effect**:
  - Higher (e.g., 0.5) → Only high-quality images accepted, may skip some detections
  - Lower (e.g., 0.1) → More permissive, accepts lower quality images
- **Note**: Set to 0.0 for violations to force feature extraction even from poor quality images

### Feature Dimension
- **Value**: `128D` (128-dimensional feature vectors)
- **Location**: `detector/layer2_reid_extractor_enhanced.py` → `output_dim`
- **Purpose**: Compressed feature vector size for efficient storage and matching
- **Original**: 512D base features from OSNet, projected to 128D
- **Effect**:
  - Higher (512D) → More accurate matching, larger storage, slower comparison
  - Lower (64D) → Faster matching, less accurate, smaller storage

---

## 3. TRACKING THRESHOLDS (SORT Algorithm)

### Max Age
- **Threshold**: `30 frames` (~1 second at 30 FPS, ~10 seconds at 3 FPS)
- **Location**: `tracker/layer3_sort_tracker.py` → `max_age`
- **Purpose**: Maximum frames a track can exist without update before deletion
- **Effect**:
  - Higher → Tracks persist longer when person temporarily hidden
  - Lower → Tracks deleted faster, more ID switches

### Min Hits
- **Threshold**: `3 frames`
- **Location**: `tracker/layer3_sort_tracker.py` → `min_hits`
- **Purpose**: Minimum consecutive detections before track is confirmed
- **Effect**:
  - Higher → Fewer false tracks, slower to establish new tracks
  - Lower → Faster track creation, more false positives

### IoU Threshold
- **Threshold**: `0.3` (30% overlap)
- **Location**: `tracker/layer3_sort_tracker.py` → `iou_threshold`
- **Purpose**: Minimum Intersection over Union for matching detections to tracks
- **Effect**:
  - Higher (e.g., 0.5) → More strict matching, more ID switches
  - Lower (e.g., 0.2) → More lenient matching, potential wrong associations

---

## 4. BEHAVIOR ANALYSIS THRESHOLDS

### Motion Detection
- **Threshold**: `50 pixels` (movement distance)
- **Location**: `tracker/layer5_behavior.py` → `motion_threshold`
- **Purpose**: Minimum pixel movement to classify as "erratic motion"
- **Effect**:
  - Higher → Only large movements trigger alerts
  - Lower → More sensitive to small movements

### Low Motion
- **Threshold**: `10 pixels`
- **Location**: `tracker/layer5_behavior.py` (hardcoded in motion check)
- **Purpose**: Classify as stationary/loitering
- **Effect**: Used for loitering detection logic

### Loitering Warning
- **Threshold**: `360 frames` (6 minutes at 1 FPS, 12 seconds at 30 FPS)
- **Location**: `tracker/layer5_behavior.py` → `loitering_warning_frames`
- **Purpose**: Frames before triggering loitering WARNING status
- **Effect**:
  - Higher → Longer wait before warning
  - Lower → Faster warning alerts

### Loitering Alert
- **Threshold**: `720 frames` (12 minutes at 1 FPS, 24 seconds at 30 FPS)
- **Location**: `tracker/layer5_behavior.py` → `loitering_alert_frames`  
- **Purpose**: Frames before triggering loitering ALERT status
- **Effect**:
  - Higher → Longer wait before alert
  - Lower → Faster alert notifications

---

## 5. FRAME PROCESSING

### Sample Rate
- **Value**: `3` (process every 3rd frame)
- **Location**: `main.py` → `FrameIngestor(sample_rate=3)`
- **Purpose**: Skip frames to reduce processing load
- **Effect**:
  - Higher (e.g., 5) → Process fewer frames, faster but less smooth tracking
  - Lower (e.g., 1) → Process every frame, slower but more accurate

---

## RECOMMENDED ADJUSTMENTS

### For Better Violation Detection:
```python
# In detector/single_stage_detector.py
conf_person = 0.4   # Lower to detect more people
conf_attr = 0.6     # Higher to reduce false violation alerts
```

### For Stricter Person Matching (Fewer Re-IDs):
```python
# In tracker/violation_only_reid.py
similarity_threshold = 0.70  # Higher = more strict (current: 0.62)
```

### For More Lenient Matching (More Re-IDs):
```python
# In tracker/violation_only_reid.py
similarity_threshold = 0.50  # Lower = more lenient (was increased to 0.62)
```

### For Faster Tracking:
```python
# In tracker/layer3_sort_tracker.py
max_age = 15        # Shorter track persistence
min_hits = 2        # Faster track confirmation
```

### For Longer Loitering Detection:
```python
# In tracker/layer5_behavior.py
loitering_warning_frames = 720   # 12 minutes at 1 FPS
loitering_alert_frames = 1440    # 24 minutes at 1 FPS
```

---

## CURRENT CONFIGURATION SUMMARY

| Component | Threshold | Value | Purpose |
|-----------|-----------|-------|---------|
| **Detection** | | | |
| Person Confidence | `conf_person` | 0.5 | Detect persons |
| Mask/Helmet Confidence | `conf_attr` | 0.5 | Detect violations |
| Head Region | `head_fraction` | 0.45 | Top 45% of bbox |
| **ReID Matching** | | | |
| Feature Similarity | `similarity_threshold` | 0.62 | Match persons |
| Feature Quality | `quality_threshold` | 0.25 | Minimum quality |
| Feature Dimension | `output_dim` | 128D | Vector size |
| **Tracking** | | | |
| Max Age | `max_age` | 30 frames | Track persistence |
| Min Hits | `min_hits` | 3 frames | Track confirmation |
| IoU Matching | `iou_threshold` | 0.3 | Box overlap |
| **Behavior** | | | |
| Erratic Motion | `motion_threshold` | 50 px | Movement detection |
| Loitering Warning | `loitering_warning_frames` | 360 | 6-12 min |
| Loitering Alert | `loitering_alert_frames` | 720 | 12-24 min |

---

*Note: All thresholds can be adjusted based on your specific environment and requirements.*
