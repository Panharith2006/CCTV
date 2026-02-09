# CCTV AI Pipeline Architecture (with ReID)

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMERA FEED                               │
│                    (Video / RTSP / Webcam)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  LAYER 1: Frame Ingest                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Read frames from source                               │ │
│  │ • Sample every 3rd frame                                │ │
│  │ • Add timestamp & frame_id                              │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Frame + Metadata
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  LAYER 2: YOLO Detection                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Detect objects (person, mask, helmet)                 │ │
│  │ • Extract bounding boxes                                │ │
│  │ • Confidence scores                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Detections: [{bbox, class, conf}, ...]
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  LAYER 3: DeepSort Tracking                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Associate detections across frames                    │ │
│  │ • Assign temporary track_id                             │ │
│  │ • Kalman filter prediction                              │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Tracks: [{track_id, bbox}, ...]
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  🔥 LAYER 3.5: ReID Identity Manager (NEW!)                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Feature Extraction:                                     │ │
│  │  • Crop person from frame                               │ │
│  │  • Extract 128-D appearance vector                      │ │
│  │  • HSV color + Edge features                            │ │
│  │                                                          │ │
│  │ Identity Matching:                                      │ │
│  │  • Compare with database vectors                        │ │
│  │  • Cosine similarity > 0.75                             │ │
│  │  • Match → use existing person_id                       │ │
│  │  • No match → create new person_id                      │ │
│  │                                                          │ │
│  │ Database Storage (SQLite):                              │ │
│  │  • Store identities (person_id, vector, timestamps)     │ │
│  │  • Log sightings (camera, decision, reason)             │ │
│  │  • Mark suspects (Warning/Alert only)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Tracks: [{track_id, person_id, bbox}, ...]
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  LAYER 4: Motion Analysis                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Multi-gap motion tracking (1,5,10,15,20 frames)       │ │
│  │ • Direction changes                                     │ │
│  │ • Average speed                                         │ │
│  │ • Displacement                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Motion: [{track_id, motion_gaps, ...}, ...]
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  LAYER 5: Behavior Decision                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Uses person_id (persistent) for state tracking:        │ │
│  │                                                          │ │
│  │ Decision Logic:                                         │ │
│  │  • Standing still (time-based)                          │ │
│  │  • Disoriented navigation                               │ │
│  │  • PPE detection (mask, helmet)                         │ │
│  │                                                          │ │
│  │ Output:                                                 │ │
│  │  • Normal (blue)                                        │ │
│  │  • Warning (orange) → Mark as suspect                   │ │
│  │  • Alert (red) → Mark as suspect                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Behavior: [{person_id, decision, reason}, ...]
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  LAYER 6: Telegram Notifier                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Send alerts for Warning/Alert                         │ │
│  │ • Cooldown: 240 seconds per person_id                   │ │
│  │ • Attach snapshot image                                 │ │
│  │ • Include: camera_id, person_id, decision, reason       │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   TELEGRAM    │
                    │   BOT ALERT   │
                    └───────────────┘
```

---

## Data Flow Example

### Frame 1: Person Enters
```
Frame → YOLO → [person bbox]
     → DeepSort → track_id=7
     → ReID → Extract features → person_id=1 (NEW)
     → Motion → No history
     → Behavior → Normal
     → Display: "P1 | Normal"
```

### Frame 50: Person Still Present
```
Frame → YOLO → [person bbox]
     → DeepSort → track_id=7 (same)
     → ReID → Match features → person_id=1 (same)
     → Motion → motion_gaps={1:5, 5:23, 10:42, ...}
     → Behavior → Normal
     → Display: "P1 | Normal"
```

### Frame 100: Person Standing Still
```
Frame → YOLO → [person bbox]
     → DeepSort → track_id=7
     → ReID → Match features → person_id=1
     → Motion → motion_gaps={1:0, 5:2, 10:3, ...} (minimal movement)
     → Behavior → Warning (100s elapsed)
     → Database: Mark person_id=1 as suspect
     → Display: "P1 | Warning"
     → Telegram: Send alert (if cooldown passed)
```

### Frame 150: Person Exits
```
Frame → YOLO → [no detection]
     → DeepSort → track_id=7 lost
     → ReID → Cleanup track_to_person mapping
     → person_id=1 STAYS in database ✅
```

### Frame 200: Person Returns
```
Frame → YOLO → [person bbox]
     → DeepSort → track_id=12 (NEW track_id)
     → ReID → Match features → person_id=1 (SAME person_id!) ✅
     → Motion → Resume from last known position
     → Behavior → Check still_time (continues from before)
     → Display: "P1 | Warning" (state preserved!)
```

---

## Key Components

### 1. Track ID (Temporary)
```
┌───────────────────────────┐
│  track_id = 7             │
│                           │
│  Assigned by: DeepSort    │
│  Lifespan: While visible  │
│  Purpose: Frame-to-frame  │
│           association     │
│                           │
│  Lost when:               │
│  • Person exits frame     │
│  • Long occlusion         │
└───────────────────────────┘
```

### 2. Person ID (Persistent) 🔥
```
┌───────────────────────────┐
│  person_id = 1            │
│                           │
│  Assigned by: ReID        │
│  Lifespan: Forever        │
│  Purpose: Global identity │
│                           │
│  Survives:                │
│  ✅ Frame exits           │
│  ✅ Camera switches       │
│  ✅ Time gaps             │
│  ✅ Occlusions            │
└───────────────────────────┘
```

### 3. Feature Vector
```
┌───────────────────────────┐
│  128-D Vector             │
│                           │
│  [0.12, 0.45, 0.89, ...]  │
│                           │
│  Contains:                │
│  • HSV color (96-D)       │
│  • Edge features (32-D)   │
│                           │
│  Used for:                │
│  • Identity matching      │
│  • Similarity comparison  │
└───────────────────────────┘
```

### 4. Database Schema
```
identities table:
┌────────────┬────────────────┬────────────┬───────────┬─────────────┬────────────┐
│ person_id  │ feature_vector │ first_seen │ last_seen │ alert_count │ is_suspect │
├────────────┼────────────────┼────────────┼───────────┼─────────────┼────────────┤
│ 1          │ 0.12,0.45,...  │ 14:30:15   │ 14:35:22  │ 3           │ 1          │
│ 2          │ 0.87,0.23,...  │ 14:32:10   │ 14:32:45  │ 0           │ 0          │
└────────────┴────────────────┴────────────┴───────────┴─────────────┴────────────┘

sightings table:
┌────┬───────────┬───────────┬────────────┬──────────┬────────────────────────┐
│ id │ person_id │ camera_id │ timestamp  │ decision │ reason                 │
├────┼───────────┼───────────┼────────────┼──────────┼────────────────────────┤
│ 1  │ 1         │ CAM_01    │ 14:30:15   │ Alert    │ Standing still too long│
│ 2  │ 1         │ CAM_01    │ 14:32:10   │ Warning  │ Disoriented navigation │
└────┴───────────┴───────────┴────────────┴──────────┴────────────────────────┘
```

---

## Comparison: Before vs After ReID

### BEFORE (Track ID Only)
```
Person walks in → track_id=7 assigned
Person walks out → track_id=7 LOST ❌
Person walks in again → track_id=12 assigned (different!) ❌
Result: System thinks it's 2 different people
```

### AFTER (with ReID)
```
Person walks in → track_id=7, person_id=1 assigned
Person walks out → track_id=7 lost, person_id=1 stored ✅
Person walks in again → track_id=12, person_id=1 matched ✅
Result: System knows it's the same person
```

---

## Decision Tree

```
Frame Input
    │
    ├─→ Person Detected?
    │   ├─→ NO → Skip
    │   └─→ YES
    │       │
    │       ├─→ Track with DeepSort → track_id
    │       │
    │       ├─→ Extract ReID features → 128-D vector
    │       │
    │       ├─→ Compare with database
    │       │   ├─→ Similarity > 0.75? → Use existing person_id
    │       │   └─→ Similarity < 0.75? → Create new person_id
    │       │
    │       ├─→ Analyze motion → motion_gaps, direction_changes
    │       │
    │       └─→ Decide behavior
    │           ├─→ Normal → Display only (no DB write)
    │           ├─→ Warning → Mark as suspect + Telegram
    │           └─→ Alert → Mark as suspect + Telegram
    │
    └─→ Continue to next frame
```

---

## Performance Metrics

```
Pipeline Component     CPU Usage    Memory      Latency
─────────────────────────────────────────────────────────
Frame Ingest (L1)      2-3%         ~10MB       5ms
YOLO Detection (L2)    15-20%       ~200MB      50ms
DeepSort (L3)          5-8%         ~50MB       15ms
🔥 ReID (L3.5)         5-10%        ~50MB       15ms
Motion Analysis (L4)   1-2%         ~20MB       5ms
Behavior Logic (L5)    <1%          ~5MB        2ms
Telegram (L6)          <1%          ~5MB        100ms*
─────────────────────────────────────────────────────────
TOTAL                  ~30-45%      ~340MB      ~190ms/frame

* Telegram latency only when sending (not every frame)
```

---

## Memory Layout

```
RAM Usage Breakdown:

YOLO Model          ~200MB  ████████████████████
DeepSort Embeddings ~50MB   █████
ReID Cache          ~50MB   █████
Motion History      ~20MB   ██
Frame Buffer        ~10MB   █
Other               ~10MB   █
                    ─────────────────────────
TOTAL:             ~340MB
```

---

## File System

```
Disk Usage:

yolov8n.pt          6.3MB   (YOLO weights)
identity.db         ~1KB    (empty database)
                    +10KB   (per 100 suspects)
snapshots/          Variable (alert images)
```

---

## Threading Model

```
┌─────────────────────────────────────────┐
│  Main Thread (Sequential)               │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ 1. Frame Capture                  │ │
│  └───────────────────────────────────┘ │
│           ↓                             │
│  ┌───────────────────────────────────┐ │
│  │ 2. YOLO Detection                 │ │
│  └───────────────────────────────────┘ │
│           ↓                             │
│  ┌───────────────────────────────────┐ │
│  │ 3. DeepSort Tracking              │ │
│  └───────────────────────────────────┘ │
│           ↓                             │
│  ┌───────────────────────────────────┐ │
│  │ 4. 🔥 ReID Processing             │ │
│  └───────────────────────────────────┘ │
│           ↓                             │
│  ┌───────────────────────────────────┐ │
│  │ 5. Motion Analysis                │ │
│  └───────────────────────────────────┘ │
│           ↓                             │
│  ┌───────────────────────────────────┐ │
│  │ 6. Behavior Decision              │ │
│  └───────────────────────────────────┘ │
│           ↓                             │
│  ┌───────────────────────────────────┐ │
│  │ 7. Telegram Alert (async)         │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘

Note: Currently single-threaded for simplicity.
      Can be parallelized for better performance.
```

---

## 🎯 Key Takeaways

1. **ReID is Layer 3.5** - Sits between tracking and motion
2. **person_id is persistent** - Survives frame exits
3. **Only suspects stored** - Database contains Warning/Alert only
4. **128-D vectors** - Appearance representation
5. **Cosine similarity** - Matching algorithm
6. **SQLite database** - Lightweight storage
7. **Zero breaking changes** - All existing code still works

---

**Architecture Version:** 2.0 with ReID  
**Last Updated:** January 29, 2026
