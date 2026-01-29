# Person Re-Identification Management Guide

## ✅ What's Implemented

Your system now has complete semi-automatic person identification with:

1. **Automatic thumbnail saving** - Every new person gets a thumbnail saved
2. **Name labeling** - Assign names like "Rith" to person IDs  
3. **Manual enrollment** - Add known people before they appear
4. **Duplicate detection** - Find similar person IDs
5. **Database management** - Merge, delete, review persons

## 📁 New Files Created

- `database/reid_database.py` - Updated with thumbnail_path and name columns
- `tracker/layer3_reid_manager.py` - Auto-saves thumbnails when creating persons
- `tools/enroll_person.py` - Manual enrollment tool
- `tools/review_persons.py` - Database management CLI
- `thumbnails/` - Directory where person crops are saved

## 🚀 How to Use

### 1. Manual Enrollment (Recommended First Step)

Enroll yourself (e.g., "Rith") before starting the main system:

```bash
python -m tools.enroll_person
```

**Options:**
- **Webcam**: Stand in front of camera, press SPACE to capture
- **Image file**: Load an existing photo

**Example session:**
```
Options:
  1. Enroll from webcam
  2. Enroll from image file
  
Select option (1-2): 1
Enter person's name (e.g., 'Rith'): Rith

[Instructions appear]
Press SPACE to capture...

✓ Successfully enrolled!
  Person ID: 1
  Name: Rith
  Thumbnail: thumbnails/enroll_Rith.jpg
```

### 2. Run Main System

Now when you run the system, it will recognize you:

```bash
python main.py
```

**What you'll see:**
- First time: `P001` (Person 1) - your enrolled identity
- When you leave and return: Still `P001` (not a new ID!)
- Different person: `P002`, `P003`, etc.

### 3. Review and Clean Database

After running for a while, review stored persons:

```bash
python -m tools.review_persons
```

**Options:**
```
1. List all persons          - See all person IDs and names
2. Find similar persons      - Detect potential duplicates
3. Merge persons            - Combine duplicate IDs
4. Label person             - Add names to unnamed IDs
5. Delete person            - Remove bad registrations
6. View thumbnails          - Browse saved person crops
```

**Example: Labeling unnamed persons**
```
Select option: 4

ID    Name         Appearances  First Seen
---   ----------   -----------  ----------
1     Rith         45          2026-01-29 14:30:00
2     (unnamed)    12          2026-01-29 14:35:00
3     (unnamed)    8           2026-01-29 14:40:00

Enter person ID to label: 2
Enter person's name: Friend1

✓ Person 2 labeled as: Friend1
```

**Example: Merging duplicates**
```
Select option: 2
Finding similar persons...

Person 1    Person 3    Similarity
--------    --------    ----------
1           3           0.856

Found 1 potentially duplicate pairs

Select option: 3
Enter source person ID to merge: 3
Enter target person ID (keep this one): 1

⚠️  Merge Person 3 into Person 1? (yes/no): yes

✓ Successfully merged Person 3 into Person 1
```

### 4. Check Stored Data

View thumbnails and person info:

```bash
# List all thumbnails
ls thumbnails/

# Expected files:
# person_001_1706526123456.jpg  <- Auto-saved when Person 1 registered
# person_002_1706526234567.jpg
# enroll_Rith.jpg               <- Manually enrolled
```

## 🎯 Best Practices

### For Accurate Recognition

1. **Enroll yourself first** with `tools/enroll_person.py`
2. **Use good lighting** when enrolling
3. **Capture multiple angles** (optional: run enrollment multiple times)
4. **Label persons** using `tools/review_persons.py` after each session
5. **Merge duplicates** weekly to keep database clean

### Database Maintenance Schedule

- **Daily**: Label unnamed persons after each recording session
- **Weekly**: Find and merge similar persons (duplicates)
- **Monthly**: Delete bad registrations (wrong crops, partial faces)

## 🔧 Configuration

### Adjust Sensitivity (if needed)

Edit `main.py` or `tracker/layer3_reid_manager.py`:

```python
# Lower threshold = more lenient matching (fewer duplicates)
reid_manager = ReIDManager(
    db_path="cctv_reid.db", 
    similarity_threshold=0.65,  # Default: 0.7, try 0.65-0.75
    thumbnail_dir="thumbnails"
)

# Require more frames before creating new person
reid_manager.required_new_frames = 3  # Default: 2, try 3-4
```

### Change Database Location

```python
# Use different DB for testing vs production
reid_manager = ReIDManager(db_path="test_reid.db")  # Testing
reid_manager = ReIDManager(db_path="cctv_reid.db")  # Production
```

## 📊 Understanding Person IDs vs Track IDs

**Track ID** (ephemeral):
- Assigned by SORT tracker
- Changes when person leaves and returns
- Example: `T3` → `T5` (same person, different track)

**Person ID** (persistent):
- Assigned by ReID system
- Stays same across appearances
- Example: `P001` (Rith) - always the same

**Display format in UI:**
```
P001 | Normal        ← Person 1 (Rith), no issues
P002 | Warning       ← Person 2 (unnamed), mask detected
T5   | Normal        ← Unknown person (pending registration)
```

## 🐛 Troubleshooting

### Same person gets multiple IDs

**Solutions:**
1. Run `python -m tools.review_persons` → option 2 (find similar)
2. Merge duplicates with option 3
3. Lower similarity threshold to 0.65
4. Increase `required_new_frames` to 3 or 4

### No thumbnails appearing

**Check:**
```bash
# Verify thumbnail directory exists
ls thumbnails/

# If empty, manually create and run again
mkdir thumbnails
python main.py
```

### Cannot find person in database

**Check database contents:**
```bash
python -m tools.review_persons
# Select option 1 to list all persons
```

## 💡 Advanced Tips

### Export Person Registry

```bash
python - <<'PY'
from database.reid_database import ReIDDatabase
db = ReIDDatabase('cctv_reid.db')
persons = db.get_all_persons()
for p in persons:
    pid, first, last, count, name, thumb = p
    print(f"{pid}: {name or 'unnamed'} ({count} appearances)")
PY
```

### Backup Database

```bash
# Copy database file
cp cctv_reid.db cctv_reid_backup_$(date +%Y%m%d).db

# Copy thumbnails
cp -r thumbnails thumbnails_backup_$(date +%Y%m%d)
```

### Reset Database (Start Fresh)

```bash
# Delete old database
rm cctv_reid.db test_reid.db

# Clear thumbnails
rm -rf thumbnails/*

# Enroll yourself again
python -m tools.enroll_person
```

## 📝 Quick Command Reference

```bash
# Enroll yourself
python -m tools.enroll_person

# Run main system
python main.py

# Review and manage persons
python -m tools.review_persons

# Test ReID pipeline
python -m test.test_reid_full

# Verify project
python verify_project.py
```

## 🎓 For Your Final Year Project Report

Document the following:

1. **Semi-automatic enrollment** - Shows practical deployment thinking
2. **Database management tools** - Demonstrates software engineering skills
3. **Thumbnail storage** - Visual verification and debugging
4. **Privacy considerations** - Store vectors, optional thumbnails, consent
5. **Maintenance workflow** - Label → merge → cleanup cycle

This makes your project production-ready, not just a prototype!

---

**Status**: ✅ Fully Implemented  
**Next**: Run `python -m tools.enroll_person` and enroll yourself as "Rith"!
