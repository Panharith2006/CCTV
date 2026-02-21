# VIOLATION-ONLY TRACKING SYSTEM

## System Architecture

### 1. Detection Workflow
```
Frame Input
    ↓
[Step 1] Detect Persons (full frame)
    ↓
[Step 2] Track Persons (SORT tracker → assign track_id)
    ↓
[Step 3] Extract Head Region (top 45% of person bbox)
    ↓
[Step 4] Detect Mask/Helmet (on head crops)
    ↓
[Step 5] Apply Priority Rules
    - Helmet → ALERT
    - Mask → WARNING
    ↓
[Step 6] Identity Management
    - Normal (no violations) → T### only (not saved)
    - Violators (mask/helmet) → P### (saved & re-identified)
```

### 2. Model Usage
**Single Model**: `best.pt` with 3 classes
- Stage 1: Full frame → Detect persons
- Stage 2: Head crops → Detect mask/helmet

### 3. Identity Management

#### Normal Persons (No Mask/Helmet):
- Track ID only: `T001`, `T002`, `T003`...
- NOT saved to database
- Forgotten when they exit
- No re-identification

#### Violators (Mask OR Helmet):
- Person ID: `P001`, `P002`, `P003`...
- Saved to database with:
  - Feature vectors (128D/512D embeddings)
  - Thumbnail image
  - Violation type (MASK/HELMET/HELMET+MASK)
  - Violation status (WARNING/ALERT)
  - First seen / Last seen timestamps
- Re-identified on return: `P001 [RE-ID]`

### 4. Re-Identification Logic

```python
# When person with violation detected:

1. Extract features from face/body
2. Compare with existing violators in database (same-day only)
3. If similarity >= threshold (0.55):
   → Reuse existing person_id
   → Update database (last_seen, appearance_count)
   → Mark as re-identified
4. Else:
   → Create new person_id
   → Save to database
   → Take thumbnail
```

### 5. Priority Rules

```
Priority 1: Helmet detected → ALERT (Red)
Priority 2: Mask detected → WARNING (Orange)
Priority 3: Loitering > 12 min → ALERT
Priority 4: Loitering > 6 min → WARNING
Priority 5: Erratic motion → WARNING
Priority 6: Normal → NORMAL (Blue)
```

### 6. Database Schema

```sql
-- Violators only
persons (
    person_id INT PRIMARY KEY,
    violation_status VARCHAR(20),      -- ALERT/WARNING
    violation_reason VARCHAR(255),     -- HELMET/MASK/HELMET+MASK
    first_seen DATETIME,
    last_seen DATETIME,
    appearance_count INT,
    is_reidentified BOOLEAN,
    thumbnail_path VARCHAR(512),
    detection_date DATE                -- For same-day filtering
)

-- Feature vectors for matching
features (
    feature_id INT PRIMARY KEY,
    person_id INT,
    feature_vector JSON,               -- 128D/512D embeddings
    quality_score FLOAT,
    timestamp DATETIME
)

-- Location tracking
location_history (
    person_id INT,
    camera_id VARCHAR(100),
    location VARCHAR(255),
    event_type VARCHAR(20),            -- entry/exit
    timestamp DATETIME
)
```

## Key Components

### Files Created:
1. **detector/single_stage_detector.py**
   - `_detect_persons()` - Detect persons on full frame
   - `_detect_attributes_on_heads()` - Detect mask/helmet on head crops

2. **tracker/violation_only_reid.py**
   - `ViolationOnlyReIDManager` - Simplified ReID for violators only
   - `identify_person()` - Returns None for normal, person_id for violators
   - `update_tracks()` - Updates tracking state

3. **main.py** (Updated)
   - Uses ViolationOnlyReIDManager
   - Simplified display logic
   - Clean statistics tracking

### Files Modified:
- `SINGLE_STAGE_DETECTOR.md` - Updated documentation

## Configuration

```python
# Detection thresholds
CONF_THRESHOLD_PERSON = 0.5        # Person detection
CONF_THRESHOLD_MASK = 0.65         # Mask detection
CONF_THRESHOLD_HELMET = 0.65       # Helmet detection
HEAD_FRAC = 0.45                   # Head region (top 45%)

# Temporal smoothing (reduce false positives)
ATTRIBUTE_SMOOTHING = 5            # Window size
ATTRIBUTE_REQUIRED_MASK = 2        # Need 2/5 frames
ATTRIBUTE_REQUIRED_HELMET = 3      # Need 3/5 frames

# Re-identification
similarity_threshold = 0.55        # Match threshold
```

## Statistics Tracked

```
- Total frames processed
- Total persons detected
- Total violators tracked (saved)
- Total normal persons (ignored)
- Alerts sent (HELMET, severe loitering)
- Warnings logged (MASK, moderate loitering)
```

## Display Format

```
Normal person:  [T001 | NORMAL]      (Blue box)
Violator:       [P001 | WARNING]     (Orange box)
Alert:          [P002 | ALERT]       (Red box)
Re-identified:  [P001 [RE-ID] | ALERT] (Red box)
```

## Benefits

1. **Focused Database**: Only violations stored
2. **Better Detection**: Head crops improve mask/helmet accuracy
3. **Clear Priority**: Helmet=Alert, Mask=Warning
4. **Efficient ReID**: Only match violators
5. **Reduced Clutter**: Normal persons not saved
6. **Single Model**: One model does everything

## Usage

```bash
# Ensure best.pt exists with person, mask, helmet classes
python main.py
```

Expected output:
```
[INFO] Loading trained model: best.pt
[Detector] Targeted detection workflow enabled
[Detector] Strategy: Person detection → Head crop → Mask/Helmet detection
[ViolationReID] Strategy: VIOLATION-ONLY (mask/helmet → database, normal → ignored)
[INFO] CCTV pipeline started
[INFO] Detection strategy: Person → Head crop → Mask/Helmet
[INFO] Storage strategy: Violators ONLY (mask/helmet → database)
```

## Testing

1. Person without mask/helmet:
   - Shows: `T001 | NORMAL`
   - Database: NOT saved
   
2. Person with mask:
   - Shows: `P001 | WARNING`
   - Database: Saved with violation_reason="MASK"
   
3. Same person returns:
   - Shows: `P001 [RE-ID] | WARNING`
   - Database: Updated (appearance_count++, last_seen)
   
4. Person with helmet:
   - Shows: `P002 | ALERT`
   - Database: Saved with violation_reason="HELMET"
   - Telegram: Alert sent (if configured)
