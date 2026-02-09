# QUICK VERIFICATION ANSWERS

## ✅ 1. Feature Extraction for ALL Persons

**YES** - Features are extracted for EVERY person (normal + violation).

**Code:** [layer3_reid_manager.py#L139](tracker/layer3_reid_manager.py)
```python
# Happens BEFORE violation check
features = self.reid_extractor.extract_features(frame, bbox)
```

**Flow:**
- Normal person → Features extracted → Compared → Memory only (not saved)
- Violation person → Features extracted → Compared → Database (saved)

**Why:** Enables "suspect removed mask later" re-identification.

---

## ✅ 2. is_reidentified Definition (ONE SENTENCE)

**Definition:**
> `is_reidentified = TRUE` means this person was in the database from a PREVIOUS tracking session and has now RETURNED (not continuously tracked).

**Code:** [layer3_reid_manager.py#L206-L215](tracker/layer3_reid_manager.py)
```python
# Check if currently being tracked
is_currently_tracked = any(
    track_data['person_id'] == best_match_id 
    for track_data in self.active_tracks.values()
)

# TRUE = person left and came back
is_reidentified = not is_currently_tracked
```

**Examples:**
- Person saved 2 seconds ago, still in view → `FALSE` (continuous)
- Person left 30 mins ago, now returns → `TRUE` (re-identified)

**Telegram:**
- `FALSE` → "NEW SUSPECT" or "MATCHED (CONTINUOUS)"
- `TRUE` → "RE-IDENTIFIED (RETURNED)" + "Known violator returned!"

---

## ✅ 3. Location History Storage (EVENT-BASED)

**YES** - Storage only on meaningful events (camera change), NOT every frame.

**Code:** [reid_database.py#L413-L420](database/reid_database.py)
```python
# Check if camera changed
if not last_camera or last_camera[0] != camera_id:
    # Only then add location history entry
    cursor.execute("""INSERT INTO location_history ...""")
```

**Result:**
- 5 minutes tracking @ 30fps = ~9,000 frames
- Per-frame storage = 9,000 DB writes ❌
- Event-based storage = 3-4 DB writes ✅

**Event-Based Principle:** Database stores EVENTS, not states.

---

## 📝 Summary

| Point | Answer | Impact |
|-------|--------|--------|
| All persons get features? | ✅ YES | Enables full re-identification |
| is_reidentified meaning? | ✅ CLEAR | Person returned after leaving |
| Event-based storage? | ✅ YES | Clean DB, no noise |

**Status:** All implemented correctly ✅

**Details:** See [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) for full documentation.
