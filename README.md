# CCTV AI Person Re-Identification System 🚀

## 📊 Project Status Report

### ✅ COMPLETED (Steps 1-3)
You've made **excellent progress**! Here's what you have:

#### ✅ STEP 1: Minimum Working System Defined
- ✅ Clear goal: Multi-camera CCTV with person detection + behavior analysis
- ✅ Mask/Helmet detection for abnormal behavior
- ✅ Motion tracking and loitering detection
- ✅ Telegram notifications

#### ✅ STEP 2: Environment Setup
- ✅ Python environment ready
- ✅ PyTorch + Ultralytics YOLO
- ✅ OpenCV for video processing
- ✅ All dependencies in `requirements.txt`

#### ✅ STEP 3: Person Detection (WORKING!)
- ✅ YOLOv8 integration ([layer2_yolo_detector.py](detector/layer2_yolo_detector.py))
- ✅ Two-stage detector (person + attributes)
- ✅ Bounding box extraction working
- ✅ Test scripts created ([test_layer1.py](test/test_layer1.py), [test_layer2.py](test/test_layer2.py))
- ✅ Frame ingestion pipeline ([layer1_frame_ingest.py](ingest/layer1_frame_ingest.py))
- ✅ SORT tracking ([layer3_sort_tracker.py](tracker/layer3_sort_tracker.py))
- ✅ Motion analysis ([layer4_motion_tracker.py](tracker/layer4_motion_tracker.py))
- ✅ Behavior rules ([layer5_behavior.py](tracker/layer5_behavior.py))
- ✅ Full pipeline in [main.py](main.py)

### ❌ MISSING (Steps 4-6) - **THIS IS YOUR NEXT FOCUS**

#### ❌ STEP 4: Person Re-Identification (128-D Vector) ⚠️ **CRITICAL MISSING**
- ❌ **No ReID model integrated**
- ❌ No feature extraction (128-D vectors)
- ❌ No person embedding generation
- ❌ Cannot distinguish between different people
- ❌ Track IDs reset when person leaves frame

**Current limitation**: Your SORT tracker assigns IDs, but when a person leaves and returns, they get a NEW ID. ReID solves this!

#### ❌ STEP 5: Database (Vector Storage) ⚠️ **CRITICAL MISSING**
- ❌ No SQLite/database integration
- ❌ No persistent person storage
- ❌ Cannot remember people across sessions
- ❌ No historical tracking

#### ❌ STEP 6: Similarity Matching ⚠️ **CRITICAL MISSING**
- ❌ No cosine similarity comparison
- ❌ Cannot re-identify returning persons
- ❌ No cross-camera identification

---

## 🎯 WHAT TO DO NEXT (Your Immediate Tasks)

### 🚨 PRIORITY 1: Add Person Re-Identification (STEP 4)

This is the **CORE** of your final year project! Without ReID, you only have a basic tracker.

#### Task 4.1: Install ReID Model
```bash
pip install torchreid
```

#### Task 4.2: Create ReID Feature Extractor
Create new file: `detector/layer2_reid_extractor.py`

```python
import torch
import torchreid
import cv2
import numpy as np

class ReIDExtractor:
    def __init__(self, model_name='osnet_x1_0'):
        """
        Initialize ReID model for feature extraction
        model_name options: 'osnet_x1_0', 'osnet_x0_75', 'osnet_ain_x1_0'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ReID] Using device: {self.device}")
        
        # Load pretrained model
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,  # dummy, we only need features
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"[ReID] Loaded model: {model_name}")

    def extract_features(self, frame, bbox):
        """
        Extract 128-D feature vector from person crop
        
        Args:
            frame: Full frame (numpy array)
            bbox: [x1, y1, x2, y2]
        
        Returns:
            feature: 128-D numpy array
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop person from frame
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # Resize to 256x128 (standard ReID input)
        person_crop = cv2.resize(person_crop, (128, 256))
        
        # Normalize (ImageNet stats)
        person_crop = person_crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        person_crop = (person_crop - mean) / std
        
        # Convert to tensor (C, H, W)
        person_tensor = torch.from_numpy(person_crop).permute(2, 0, 1).unsqueeze(0)
        person_tensor = person_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(person_tensor)
            features = features.cpu().numpy().flatten()
        
        # Normalize to unit vector
        features = features / (np.linalg.norm(features) + 1e-12)
        
        print(f"[ReID] Extracted feature shape: {features.shape}")
        return features

    def compare_features(self, feat1, feat2):
        """
        Compute cosine similarity between two feature vectors
        Returns similarity score (0 to 1, higher = more similar)
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        similarity = np.dot(feat1, feat2)
        return float(similarity)
```

#### Task 4.3: Create Test Script
Create new file: `test/test_reid.py`

```python
import cv2
import numpy as np
from detector.layer2_reid_extractor import ReIDExtractor
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from config.layer0_cameras import CAMERAS

def main():
    print("=== Testing ReID Feature Extraction ===\n")
    
    # Initialize components
    cam = CAMERAS[0]
    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )
    
    detector = YOLODetector(classes=["person"], conf_threshold=0.5)
    reid_extractor = ReIDExtractor()
    
    # Store first person's features
    reference_features = None
    reference_bbox = None
    
    print("\n[INFO] Detecting first person as reference...")
    
    for data in ingestor.read():
        frame = data["frame"]
        detections = detector.detect(frame)
        
        # Filter person detections
        persons = [d for d in detections if d["class"] == "person"]
        
        if len(persons) == 0:
            cv2.putText(frame, "No person detected - move into view", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("ReID Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        # Use first detection as reference
        if reference_features is None:
            person = persons[0]
            reference_bbox = person["bbox"]
            reference_features = reid_extractor.extract_features(frame, reference_bbox)
            
            if reference_features is not None:
                print(f"\n✓ Reference person captured!")
                print(f"  Feature vector shape: {reference_features.shape}")
                print(f"  Feature vector (first 10 dims): {reference_features[:10]}")
                print(f"\n[INFO] Now detecting and comparing all persons...\n")
        
        # Compare all detected persons with reference
        for i, person in enumerate(persons):
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Extract features
            features = reid_extractor.extract_features(frame, bbox)
            
            if features is not None and reference_features is not None:
                # Compute similarity
                similarity = reid_extractor.compare_features(reference_features, features)
                
                # Determine if same person (threshold = 0.7)
                is_same = similarity > 0.7
                color = (0, 255, 0) if is_same else (0, 0, 255)
                label = f"Person {i+1} | Sim: {similarity:.3f} | {'MATCH' if is_same else 'DIFFERENT'}"
                
                print(f"  Person {i+1}: Similarity = {similarity:.3f} ({'SAME' if is_same else 'DIFFERENT'})")
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display reference bbox
        if reference_bbox is not None:
            rx1, ry1, rx2, ry2 = reference_bbox
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
            cv2.putText(frame, "REFERENCE", (rx1, ry1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(frame, "ESC to exit | Green=Match, Red=Different, Blue=Reference", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("ReID Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    ingestor.release()
    cv2.destroyAllWindows()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
```

#### Task 4.4: Test ReID Extraction
```bash
python test/test_reid.py
```

**What you should see:**
- First person detected becomes reference (blue box)
- Same person gets green box + high similarity (>0.7)
- Different person gets red box + low similarity (<0.7)
- Console prints feature vector shape: (512,) or (2048,) depending on model

**If it works, CELEBRATE! 🎉 You now have the CORE technology!**

---

### 🚨 PRIORITY 2: Add Database Storage (STEP 5)

#### Task 5.1: Create Database Schema
Create new file: `database/reid_database.py`

```python
import sqlite3
import numpy as np
import json
from datetime import datetime

class ReIDDatabase:
    def __init__(self, db_path="reid_database.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        print(f"[Database] Connected to {db_path}")
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Person table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen DATETIME,
                last_seen DATETIME,
                appearance_count INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Feature vectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                feature_vector TEXT,
                timestamp DATETIME,
                camera_id TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)
        
        # Detections table (for audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                camera_id TEXT,
                bbox TEXT,
                timestamp DATETIME,
                track_id INTEGER,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)
        
        self.conn.commit()
        print("[Database] Tables created")
    
    def add_person(self, feature_vector, camera_id):
        """Add new person to database"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Insert person
        cursor.execute("""
            INSERT INTO persons (first_seen, last_seen)
            VALUES (?, ?)
        """, (now, now))
        person_id = cursor.lastrowid
        
        # Insert feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id)
            VALUES (?, ?, ?, ?)
        """, (person_id, feature_json, now, camera_id))
        
        self.conn.commit()
        print(f"[Database] Added new person: ID={person_id}")
        return person_id
    
    def get_all_features(self):
        """Get all person features for matching"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_id, feature_vector
            FROM features
            WHERE person_id IN (SELECT person_id FROM persons WHERE status='active')
        """)
        
        results = []
        for row in cursor.fetchall():
            person_id = row[0]
            feature_vector = np.array(json.loads(row[1]))
            results.append({
                'person_id': person_id,
                'features': feature_vector
            })
        
        return results
    
    def update_person(self, person_id, feature_vector, camera_id):
        """Update person's last seen and add new feature"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Update person
        cursor.execute("""
            UPDATE persons
            SET last_seen = ?, appearance_count = appearance_count + 1
            WHERE person_id = ?
        """, (now, person_id))
        
        # Add new feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id)
            VALUES (?, ?, ?, ?)
        """, (person_id, feature_json, now, camera_id))
        
        self.conn.commit()
    
    def add_detection(self, person_id, camera_id, bbox, track_id):
        """Log a detection"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (person_id, camera_id, bbox, timestamp, track_id)
            VALUES (?, ?, ?, ?, ?)
        """, (person_id, camera_id, json.dumps(bbox), datetime.now(), track_id))
        self.conn.commit()
    
    def get_person_stats(self, person_id):
        """Get statistics for a person"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT first_seen, last_seen, appearance_count
            FROM persons
            WHERE person_id = ?
        """, (person_id,))
        return cursor.fetchone()
    
    def close(self):
        self.conn.close()
```

---

### 🚨 PRIORITY 3: Add Re-Identification Logic (STEP 6)

#### Task 6.1: Create ReID Manager
Create new file: `tracker/layer3_reid_manager.py`

```python
import numpy as np
from database.reid_database import ReIDDatabase
from detector.layer2_reid_extractor import ReIDExtractor

class ReIDManager:
    def __init__(self, db_path="reid_database.db", similarity_threshold=0.7):
        """
        Manages person re-identification across frames and cameras
        
        similarity_threshold: minimum similarity to consider same person (0-1)
        """
        self.db = ReIDDatabase(db_path)
        self.reid_extractor = ReIDExtractor()
        self.similarity_threshold = similarity_threshold
        
        print(f"[ReIDManager] Initialized with threshold={similarity_threshold}")
    
    def identify_person(self, frame, bbox, camera_id):
        """
        Identify person from frame crop
        Returns: person_id (new or matched)
        """
        # Extract features
        features = self.reid_extractor.extract_features(frame, bbox)
        
        if features is None:
            return None
        
        # Get all known persons
        known_persons = self.db.get_all_features()
        
        if len(known_persons) == 0:
            # First person ever
            person_id = self.db.add_person(features, camera_id)
            print(f"[ReIDManager] New person registered: ID={person_id}")
            return person_id
        
        # Compare with all known persons
        best_match_id = None
        best_similarity = 0.0
        
        for person in known_persons:
            similarity = self.reid_extractor.compare_features(features, person['features'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person['person_id']
        
        # Decide if match or new person
        if best_similarity >= self.similarity_threshold:
            # Match found
            self.db.update_person(best_match_id, features, camera_id)
            print(f"[ReIDManager] Matched person: ID={best_match_id} (similarity={best_similarity:.3f})")
            return best_match_id
        else:
            # New person
            person_id = self.db.add_person(features, camera_id)
            print(f"[ReIDManager] New person registered: ID={person_id} (best_sim={best_similarity:.3f})")
            return person_id
    
    def update_tracking(self, tracks, frame, camera_id):
        """
        Update all tracks with ReID person IDs
        
        Args:
            tracks: list of track dicts from SORT
            frame: current frame
            camera_id: camera identifier
        
        Returns:
            tracks with added 'person_id' field
        """
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            
            # Identify person
            person_id = self.identify_person(frame, bbox, camera_id)
            
            if person_id is not None:
                track['person_id'] = person_id
                
                # Log detection
                self.db.add_detection(person_id, camera_id, bbox, track_id)
            else:
                track['person_id'] = None
        
        return tracks
    
    def close(self):
        self.db.close()
```

#### Task 6.2: Create Integration Test
Create new file: `test/test_reid_full.py`

```python
import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer3_reid_manager import ReIDManager

def main():
    print("=== Full ReID Pipeline Test ===\n")
    
    # Initialize
    cam = CAMERAS[0]
    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )
    
    detector = YOLODetector(classes=["person"], conf_threshold=0.5)
    tracker = SortTracker()
    reid_manager = ReIDManager(db_path="test_reid.db", similarity_threshold=0.7)
    
    print("\n[INFO] System ready!")
    print("[INFO] Walk in and out of frame to test re-identification")
    print("[INFO] Each unique person should get a consistent Person ID\n")
    
    frame_count = 0
    
    for data in ingestor.read():
        frame = data["frame"]
        frame_count += 1
        
        # Detect persons
        detections = detector.detect(frame)
        persons = [d for d in detections if d["class"] == "person"]
        
        # Track
        tracks = tracker.update(persons, frame)
        
        # ReID
        tracks = reid_manager.update_tracking(tracks, frame, cam["camera_id"])
        
        # Visualize
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            person_id = track.get('person_id', 'Unknown')
            
            # Color based on person_id
            if person_id != 'Unknown':
                color_idx = person_id % 10
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), 
                         (255,0,255), (0,255,255), (128,0,128), (255,128,0),
                         (128,255,0), (0,128,255)]
                color = colors[color_idx]
            else:
                color = (128, 128, 128)
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Person-{person_id} | Track-{track_id}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info
        cv2.putText(frame, f"Frame: {frame_count} | Persons: {len(tracks)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "ESC to exit", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Full ReID Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    reid_manager.close()
    ingestor.release()
    cv2.destroyAllWindows()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
```

#### Task 6.3: Test Full ReID Pipeline
```bash
python test/test_reid_full.py
```

**Expected behavior:**
- First time you appear: "Person-1"
- Walk out of frame and return: still "Person-1" (not Person-2!)
- Different person appears: gets "Person-2"
- Track IDs may change, but Person IDs stay consistent!

**If this works, your project is 90% complete! 🎉**

---

## 🏆 FINAL INTEGRATION (Task 7)

### Task 7.1: Update main.py
Add ReID to your existing pipeline:

```python
# Add at top of main.py
from tracker.layer3_reid_manager import ReIDManager

# In main() function, after tracker initialization:
reid_manager = ReIDManager(db_path="cctv_reid.db", similarity_threshold=0.7)

# In the main loop, after tracker.update():
tracks = reid_manager.update_tracking(tracks, frame, cam["camera_id"])

# In visualization, update the label:
label = f"P{track.get('person_id', '?')}-T{track_id}"
```

### Task 7.2: Run Full System
```bash
python main.py
```

---

## 📈 Project Completion Checklist

### Core Features (Must Have) ✅
- [x] Person detection (YOLO)
- [x] Tracking (SORT)
- [x] Motion analysis
- [x] Behavior rules (mask/helmet/loitering)
- [x] Telegram notifications
- [ ] **ReID feature extraction** ⚠️ **PRIORITY**
- [ ] **Database storage** ⚠️ **PRIORITY**
- [ ] **Person re-identification** ⚠️ **PRIORITY**

### Advanced Features (Nice to Have) 🎁
- [x] Multi-camera support (架构已准备好)
- [x] Mask detection (dataset + training)
- [x] Two-stage detection
- [ ] FAISS for fast vector search (optional)
- [ ] Web dashboard (optional)
- [ ] Historical playback (optional)

---

## 📊 Recommended Timeline

### Week 1 (NOW!)
- **Day 1-2**: Implement ReID extractor (Task 4.1-4.4)
- **Day 3**: Test ReID thoroughly, adjust threshold
- **Day 4-5**: Implement database (Task 5.1)
- **Day 6-7**: Implement ReID manager (Task 6.1-6.3)

### Week 2
- **Day 1-2**: Integrate ReID into main.py
- **Day 3-4**: Test with multiple people
- **Day 5**: Optimize threshold and performance
- **Day 6-7**: Document and prepare demo

### Week 3+
- Polish UI/visualization
- Add optional features
- Prepare final report
- Record demonstration video

---

## 🎓 For Your Supervisor/Report

### What Makes This Project Strong:
1. ✅ **Layered architecture** (good software engineering)
2. ✅ **Real-world application** (CCTV security)
3. ✅ **Multiple CV techniques** (detection + tracking + ReID)
4. ✅ **Complete pipeline** (ingest → detect → track → identify → alert)
5. ⏳ **Deep learning integration** (YOLO + ReID models)
6. ⏳ **Database management** (persistent storage)
7. ✅ **Rule-based AI** (behavior analysis)

### Technical Depth:
- Computer Vision (✅)
- Deep Learning (✅)
- Object Detection (✅)
- Object Tracking (✅)
- **Person Re-Identification (⏳ IN PROGRESS)**
- Database Systems (⏳ IN PROGRESS)
- Real-time Systems (✅)
- Multi-camera Systems (✅)

---

## 🆘 Troubleshooting

### If ReID is slow:
- Use `osnet_x0_75` (smaller model)
- Only extract features every 5 frames
- Use GPU if available

### If similarity is always low:
- Check image normalization
- Ensure bbox is not empty
- Try lower threshold (0.5-0.6)
- Check if person crop is too small

### If database grows too large:
- Delete old features (keep only latest 10 per person)
- Archive detections older than 30 days
- Use FAISS for faster search

---

## 📚 Resources

### ReID Papers:
- OSNet: Omni-Scale Feature Learning (2019)
- Strong Baseline for Person Re-Identification (2019)
- Deep Person Re-Identification: A Survey (2018)

### Code References:
- Torchreid: https://github.com/KaiyangZhou/deep-person-reid
- YOLO: https://github.com/ultralytics/ultralytics
- SORT: https://github.com/abewley/sort

---

## 🎯 Success Criteria

Your project will be **excellent** if:
1. ✅ Person detection works reliably
2. ✅ Tracking maintains IDs within a scene
3. ⏳ **ReID correctly identifies returning persons** ← **THIS IS THE KEY!**
4. ⏳ **Database stores and retrieves features**
5. ✅ Behavior rules trigger alerts
6. ✅ System runs in real-time

---

## 📝 Next Immediate Action

**RIGHT NOW, run these commands:**

```bash
# 1. Install ReID library
pip install torchreid

# 2. Create directory for database
mkdir database

# 3. Create the ReID extractor file (copy code from Task 4.2 above)
# Create: detector/layer2_reid_extractor.py

# 4. Create database manager (copy code from Task 5.1 above)
# Create: database/reid_database.py

# 5. Test ReID extraction
python test/test_reid.py
```

**If test_reid.py works and shows similarity scores, YOU'RE READY TO FINISH!**

---

## 💬 Questions?

Common questions:
- **Q: Is tracking same as ReID?**  
  A: No! Tracking = maintain ID within scene. ReID = recognize person who left and returned.

- **Q: Why 128-D vector?**  
  A: Compact representation of person's appearance (actually OSNet uses 512-D, even better!)

- **Q: Can I skip database?**  
  A: Not recommended. Database proves you can persist and search data (important for FYP).

- **Q: Is this enough for final year?**  
  A: **YES!** Detection + Tracking + ReID + Database = very strong project.

---

## 🚀 You're Almost There!

You've built **80%** of a professional CCTV system. The missing 20% (ReID + Database) is **the most important part** that makes this a true AI project!

**Focus on Tasks 4, 5, and 6 this week. You can do this! 💪**

Good luck! 🎓
