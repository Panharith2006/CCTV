# 🧪 ReID System Testing Guide (Single Webcam)

**Feature Vector:** 128-dimensional (changed from 512D)  
**Model:** OSNet x0.25 (lightweight, faster)

---

## 📋 Understanding the Comparison Logic

### How Person Re-Identification Works:

```python
# When a person is detected:
1. Extract 128D feature vector from body (clothes, shape, color)
2. Compare with ALL existing persons in database (using cosine similarity)
3. If similarity >= 0.7 → MATCH (same person, keep existing ID)
4. If similarity < 0.7 → NEW PERSON (assign new ID, save to database)
```

### Database Comparison Scenarios:

| Database State | New Detection | What Happens |
|----------------|---------------|--------------|
| Empty (0 persons) | Person A detected | ✅ Save as ID=1 (no comparison needed) |
| 1 person (A) | Person A again | Compare with ID=1 → Match → Keep ID=1 |
| 1 person (A) | Person B (new) | Compare with ID=1 → No match → Save as ID=2 |
| 5 persons (A-E) | Person C returns | Compare with all 5 → Match ID=3 → Keep ID=3 |
| 10 persons | Person K (new) | Compare with all 10 → No match → Save as ID=11 |

**Key Point:** Yes, every new detection compares with ALL persons in database! This is O(n) complexity but necessary for accurate Re-ID.

---

## 🎬 Test Scenarios (Single Webcam)

### 📍 Scenario 1: First Person Detection (Database Empty)

**Setup:** Start fresh (empty database)

**Test Steps:**
```bash
# 1. Clear database (optional - start fresh)
python scripts/test_mysql_connection.py

# 2. Run system
python main.py

# 3. Sit in front of webcam
```

**Expected Output:**
```
[ReIDManager] ✓ FIRST PERSON registered: ID=1
[Database] Added new person: ID=1 location=CAM_02 - Laptop Webcam
```

**What Happened:**
- No comparison (database empty)
- Extracted 128D vector
- Saved to database as Person 1
- Thumbnail saved: `thumbnails/person_001_timestamp.jpg`

---

### 📍 Scenario 2: Same Person Re-Detection (ONE ID Logic)

**Setup:** Keep system running from Scenario 1

**Test Steps:**
```bash
# 1. Stay in front of webcam
# 2. Move around (walk left/right)
# 3. Look for repeated detections
```

**Expected Output:**
```
[ReIDManager] Best match: P001 = 0.823 (threshold: 0.7)
[ReIDManager] ✓ MATCHED: Person ID=1 (even if mask changed)
```

**What Happened:**
- Extracted 128D vector
- Compared with Person 1 in database
- Similarity = 0.823 (above 0.7 threshold)
- **Kept same ID=1** (no new database entry)

**Verification:**
```bash
# Check database - should still be only 1 person
python scripts/print_db.py
```

---

### 📍 Scenario 3: New Person Detection (Different Person)

**Setup:** System running with Person 1 in database

**Test Steps:**
```bash
# 1. Ask someone else to sit in front of webcam
# OR
# 2. Change your appearance significantly:
#    - Wear different colored clothes
#    - Wear a jacket/hoodie
#    - Completely different outfit
```

**Expected Output:**
```
[ReIDManager] Best match: P001 = 0.634 (threshold: 0.7)
[Database] Added new person: ID=2 location=CAM_02 - Laptop Webcam
[ReIDManager] ✓ NEW PERSON registered: ID=2
```

**What Happened:**
- Extracted 128D vector
- Compared with Person 1
- Similarity = 0.634 (below 0.7 threshold)
- **Created NEW ID=2** (saved to database)

**Verification:**
```bash
# Check database - should now have 2 persons
python scripts/print_db.py
```

---

### 📍 Scenario 4: Mask/Helmet Change (ONE ID Maintained)

**Setup:** System running with your ID=1

**Test Steps:**
```bash
# 1. Detected without mask → ID=1
# 2. Put on a mask
# 3. System should KEEP same ID=1
# 4. Remove mask
# 5. System should STILL keep ID=1
```

**Expected Output:**
```
# Without mask
[ReIDManager] Best match: P001 = 0.789 (threshold: 0.7)
[ReIDManager] ✓ MATCHED: Person ID=1

# With mask
[DEBUG] Frame X: 1 persons, 1 masks, 0 helmets
[ReIDManager] Best match: P001 = 0.743 (threshold: 0.7)
[ReIDManager] ✓ MATCHED: Person ID=1 (even if mask changed)

# Mask removed
[ReIDManager] Best match: P001 = 0.801 (threshold: 0.7)
[ReIDManager] ✓ MATCHED: Person ID=1
```

**What Happened:**
- Body features (clothes, shape) remain similar
- Mask status changes but ID persists
- **ONE ID maintained throughout**

---

### 📍 Scenario 5: Person Returns After Leaving

**Setup:** System running

**Test Steps:**
```bash
# 1. Sit in front of webcam → Detected as ID=1
# 2. Leave frame completely (walk away)
# 3. Wait 35+ seconds (exit timeout is 30s)
# 4. Return to webcam
```

**Expected Output:**
```
# Step 1: Initial detection
[ReIDManager] ✓ MATCHED: Person ID=1

# Step 3: After 30s
[ReIDManager] 🚪 Person 1 EXITED (Track 1 lost for 31.2s)
[Database] Person 1 EXITED scene at CAM_02 - Laptop Webcam

# Step 4: Return
[ReIDManager] Best match: P001 = 0.812 (threshold: 0.7)
[ReIDManager] ✓ MATCHED: Person ID=1 (even if mask changed)
```

**What Happened:**
- System marked exit after 30s
- Location history updated with exit event
- When you return, **same ID=1 recognized** (not a new person)

**Database Check:**
```bash
python scripts/print_db.py
# Should show:
# - Person 1: status='active' (reactivated)
# - Location history: first_detection → exit → movement
```

---

### 📍 Scenario 6: Multiple People (Database Grows)

**Setup:** Clean database

**Test Steps:**
```bash
# 1. Person A sits → ID=1 (0 comparisons)
# 2. Person B sits → ID=2 (1 comparison: vs ID=1)
# 3. Person C sits → ID=3 (2 comparisons: vs ID=1, ID=2)
# 4. Person A returns → Matched as ID=1 (3 comparisons: vs all)
# 5. Person D sits → ID=4 (3 comparisons: vs ID=1,2,3)
```

**Expected Output:**
```
# Person A
[ReIDManager] ✓ FIRST PERSON registered: ID=1

# Person B (new clothes)
[ReIDManager] Best match: P001 = 0.542
[ReIDManager] ✓ NEW PERSON registered: ID=2

# Person C (different person)
[ReIDManager] Best match: P001 = 0.601
[ReIDManager] ✓ NEW PERSON registered: ID=3

# Person A returns
[ReIDManager] Best match: P001 = 0.834 (threshold: 0.7)
[ReIDManager] ✓ MATCHED: Person ID=1

# Person D (another new person)
[ReIDManager] Best match: P002 = 0.623
[ReIDManager] ✓ NEW PERSON registered: ID=4
```

**What Happened:**
- Database grows: 1 → 2 → 3 → 3 (A matched) → 4
- Each new detection compares with ALL existing persons
- System finds best match and checks threshold

**Performance Note:**
- 3 persons = 3 comparisons per detection
- 10 persons = 10 comparisons per detection
- 100 persons = 100 comparisons per detection
- Still fast with 128D vectors (~1-2ms per comparison)

---

## 🧪 How to Verify System is Working

### Method 1: Check Console Output

**Look for these key messages:**

✅ **Correct ID Persistence:** Same person keeps same ID
```
[ReIDManager] ✓ MATCHED: Person ID=1
[ReIDManager] ✓ MATCHED: Person ID=1
[ReIDManager] ✓ MATCHED: Person ID=1
```

✅ **New Person Detection:** Different person gets new ID
```
[ReIDManager] Best match: P001 = 0.621 (threshold: 0.7)
[ReIDManager] ✓ NEW PERSON registered: ID=2
```

✅ **Exit Detection:** Person leaves for >30s
```
[ReIDManager] 🚪 Person 1 EXITED (Track 1 lost for 31.4s)
```

❌ **Problem:** ID keeps changing for same person
```
[ReIDManager] ✓ NEW PERSON registered: ID=5  # Should be ID=1!
[ReIDManager] ✓ NEW PERSON registered: ID=6  # Should be ID=1!
```

---

### Method 2: Check Database

```bash
# Print all persons in database
python scripts/print_db.py
```

**Expected Output:**
```
==========================================================
  Database Contents
==========================================================

Total Persons: 3

Person ID: 1
  Name: None
  First Seen: 2026-02-05 10:23:45
  Last Seen: 2026-02-05 10:28:12
  Status: active
  Appearances: 247
  Last Location: CAM_02 - Laptop Webcam

Person ID: 2
  Name: None
  First Seen: 2026-02-05 10:25:30
  Last Seen: 2026-02-05 10:26:55
  Status: exited
  Appearances: 89
  Last Location: CAM_02 - Laptop Webcam
```

**What to Check:**
- ✅ Appearances should increase for same person (not create new IDs)
- ✅ Status: 'active' (in scene) or 'exited' (left scene)
- ✅ Last location updated when person moves

---

### Method 3: Check Thumbnails

```bash
# Check saved images
dir thumbnails

# Should see:
# thumbnails/
#   person_001_timestamp1.jpg  (Person 1 first detection)
#   person_002_timestamp2.jpg  (Person 2 first detection)
#   person_003_timestamp3.jpg  (Person 3 first detection)
```

**What to Check:**
- ✅ Only ONE thumbnail per person (first detection only)
- ✅ NOT saving every frame (controlled saving)

---

## 🎯 Test Checklist

### ✅ Basic Functionality Tests

- [ ] **Test 1:** System starts successfully (no errors)
- [ ] **Test 2:** First person detected as ID=1
- [ ] **Test 3:** Same person keeps ID=1 (not creating new IDs)
- [ ] **Test 4:** Different outfit/person creates new ID
- [ ] **Test 5:** Mask on/off does NOT change ID
- [ ] **Test 6:** Person exit detected after 30s
- [ ] **Test 7:** Person returns with same ID (not new ID)

### ✅ Database Tests

- [ ] **Test 8:** Check `python scripts/print_db.py` shows correct person count
- [ ] **Test 9:** Appearances increase for same person
- [ ] **Test 10:** Location history records movements

### ✅ Performance Tests

- [ ] **Test 11:** System runs smoothly with 1 person (~150-300ms per frame)
- [ ] **Test 12:** System handles 2-3 people simultaneously
- [ ] **Test 13:** No memory leaks (run for 5+ minutes)

---

## 🔧 Troubleshooting Tests

### Problem: Same person gets multiple IDs

**Possible Causes:**
1. Threshold too high (0.7 → try 0.65)
2. Lighting changed dramatically
3. Person changed clothes

**Test Fix:**
```bash
# Edit main.py line 119
reid_manager = ReIDManager(
    similarity_threshold=0.65  # Lower = more tolerant
)
```

---

### Problem: Different people get same ID

**Possible Causes:**
1. Threshold too low
2. People wearing similar clothes

**Test Fix:**
```bash
# Edit main.py line 119
reid_manager = ReIDManager(
    similarity_threshold=0.75  # Higher = more strict
)
```

---

### Problem: System too slow

**Current:** ~150-300ms per frame on CPU

**Speed Test:**
```python
# Add timing in main.py
import time
start = time.time()
# ... detection code ...
print(f"Frame processing: {(time.time() - start)*1000:.1f}ms")
```

**Expected:**
- CPU: 150-300ms per frame
- GPU: 30-50ms per frame (much faster)

---

## 📊 Sample Test Results

### Successful Test Example:

```
Scenario: 1 person, mask on/off, leave and return

Timeline:
00:00 - Person enters → ID=1 ✅
00:15 - Still tracking → ID=1 ✅
00:30 - Puts on mask → ID=1 ✅ (not ID=2!)
00:45 - Removes mask → ID=1 ✅
01:00 - Leaves frame
01:35 - Exit detected ✅ (after 30s)
02:00 - Returns → ID=1 ✅ (not ID=2!)

Database Check:
- Total Persons: 1 ✅
- Person 1 appearances: 280+ ✅
- Location history: 2 events (first_detection, exit) ✅
- Thumbnails: 1 image ✅

Result: PASS ✅
```

---

## 🎬 Quick Start Testing Commands

```bash
# 1. Start fresh (optional)
python scripts/test_mysql_connection.py

# 2. Run system
python main.py

# 3. Perform test scenarios (see above)
# - Sit in front of webcam
# - Move around
# - Put mask on/off
# - Leave and return

# 4. Check results
python scripts/print_db.py
dir thumbnails

# 5. Stop system
# Press Ctrl+C
```

---

## 📈 Understanding Similarity Scores

**Cosine Similarity Range:** 0.0 to 1.0

| Score | Meaning | Action |
|-------|---------|--------|
| 0.90+ | Very strong match (same person, same outfit) | ✅ Match |
| 0.75-0.89 | Strong match (same person, slight change) | ✅ Match |
| **0.70** | **Threshold** (boundary) | **Decision point** |
| 0.60-0.69 | Weak similarity (different person similar outfit) | ❌ New ID |
| 0.50-0.59 | Different (different person) | ❌ New ID |
| 0.00-0.49 | Very different (completely different) | ❌ New ID |

**Example from Console:**
```
[ReIDManager] Best match: P001 = 0.823 (threshold: 0.7)
                           ↑       ↑              ↑
                           ID    Score         Threshold
                                 ✅ > 0.7 = Match!
```

---

## 🎓 Summary

**Key Concepts to Test:**

1. ✅ **ONE ID per person** - Same person always gets same ID
2. ✅ **Mask/Helmet resilience** - ID persists even with mask changes
3. ✅ **Exit handling** - System detects when person leaves (30s timeout)
4. ✅ **Return recognition** - Person returns with same ID (not new)
5. ✅ **Controlled saving** - Only 3 save points (first, suspect, exit)
6. ✅ **Database comparison** - Every detection compares with ALL existing persons

**Testing Time:** ~15 minutes for all scenarios

**Success Criteria:** Same person consistently gets same ID, different persons get different IDs

---

**Changed to 128D vectors for faster processing!** ⚡  
Model: OSNet x0.25 (lightweight, 128-dimensional features)
