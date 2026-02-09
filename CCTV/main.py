import cv2
import time
import os
import sys
from collections import defaultdict, deque

from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.two_stage_detector import TwoStageDetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
from tracker.layer6_telegram import TelegramNotifier
from config.telegram_config import BOT_TOKEN, CHAT_ID
from tracker.layer3_reid_manager import ReIDManager


# ===================== CONFIG FLAGS =====================
# Debug levels: 0=Silent, 1=Summary only, 2=Important events, 3=Verbose (every frame)
# 
# Recommended settings:
#   DEBUG_LEVEL = 0  → Production (no console output except errors)
#   DEBUG_LEVEL = 1  → Monitoring (summary every 30 frames, ~30 seconds)
#   DEBUG_LEVEL = 2  → Events (show violations, re-identifications, alerts)
#   DEBUG_LEVEL = 3  → Debug (show all detections, attributes, motion - VERY VERBOSE)
#
DEBUG_LEVEL = 1  # <-- CHANGE THIS VALUE
OUTPUT_EVERY_N_FRAMES = 30  # Print summary every N frames (at ~1fps = 30 seconds)

# ===================== TELEGRAM COOLDOWN =====================
last_telegram_sent = {}  # (track_id, decision) -> timestamp
unknown_person_alerted = {}  # track_id -> timestamp (to track unknown person alerts)


def can_send(track_id, decision, cooldown):
    now = time.time()
    key = (track_id, decision)

    if key not in last_telegram_sent:
        last_telegram_sent[key] = now
        return True

    if now - last_telegram_sent[key] >= cooldown:
        last_telegram_sent[key] = now
        return True

    return False


# ===================== UTIL =====================
def check_bbox_overlap(person_bbox, ppe_bbox):
    px1, py1, px2, py2 = person_bbox
    dx1, dy1, dx2, dy2 = ppe_bbox

    cx = (dx1 + dx2) / 2
    cy = (dy1 + dy2) / 2

    return px1 <= cx <= px2 and py1 <= cy <= py2


# ===================== MAIN =====================
def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    # Auto-load the LATEST trained weights
    model_path = "yolov8n.pt"
    
    # First, check for best.pt in the root directory
    if os.path.exists("best.pt"):
        model_path = "best.pt"
        print(f"[INFO] Loading trained model: {model_path}")
    else:
        # Find the most recent training run
        detect_dir = os.path.join("runs", "detect")
        if os.path.exists(detect_dir):
            # Get all mask_finetune* directories sorted by modification time
            runs = [d for d in os.listdir(detect_dir) if d.startswith("mask_finetune")]
            if runs:
                # Sort by name (mask_finetune4 > mask_finetune3, etc.)
                runs.sort(reverse=True)
                for run in runs:
                    candidate = os.path.join(detect_dir, run, "weights", "best.pt")
                    if os.path.exists(candidate):
                        model_path = candidate
                        print(f"[INFO] Loading trained model: {model_path}")
                        break
    
    if model_path == "yolov8n.pt":
        print(f"[WARNING] No trained model found, using default: {model_path}")
        print(f"[WARNING] Default model doesn't detect masks! Run: python train_mask.py")

    # Tunable thresholds
    # Align person filter with detector's person confidence (was filtering out detections)
    CONF_THRESHOLD_PERSON = 0.5
    CONF_THRESHOLD_MASK = 0.65  # Increased to reduce false positives (glasses detected as masks)
    CONF_THRESHOLD_HELMET = 0.65  # Increased to reduce false positives
    HEAD_FRAC = 0.45
    IOU_THRESHOLD = 0.1
    ATTRIBUTE_SMOOTHING = 5
    ATTRIBUTE_REQUIRED_MASK = 2  # Need 2 detections in smoothing window for confirmation
    ATTRIBUTE_REQUIRED_HELMET = 3  # Need 3 detections in smoothing window for confirmation
    MOTION_THRESHOLD = 150  # Higher = less sensitive to normal movement
    LOITERING_WARNING_FRAMES = 360  # 6 minutes @ 1fps
    LOITERING_ALERT_FRAMES = 720  # 12 minutes @ 1fps

    # Two-stage detector: person detector (COCO) + attribute detector (fine-tuned)
    detector = TwoStageDetector(person_model_path="yolov8n.pt", attr_model_path=model_path, conf_person=0.5, conf_attr=0.5)
    tracker = SortTracker()
    motion = MotionAnalyzer(frame_gaps=[1, 5, 10, 15, 20])

    behavior = BehaviorDecider(
        motion_threshold=MOTION_THRESHOLD, 
        loitering_warning_frames=LOITERING_WARNING_FRAMES,
        loitering_alert_frames=LOITERING_ALERT_FRAMES,
        warning_time=100, 
        alert_time=200
    )

    # Create Telegram notifier only if credentials are set
    if BOT_TOKEN and CHAT_ID:
        telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    else:
        telegram = None

    # Attribute temporal smoothing storage
    attribute_history = defaultdict(lambda: {"mask": deque(maxlen=ATTRIBUTE_SMOOTHING), "helmet": deque(maxlen=ATTRIBUTE_SMOOTHING)})

    # Lightweight face detector for better mask association
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # ReID manager: persistent person identification across leaves/returns
    # Pass camera location for cross-camera tracking
    camera_location = f"{cam['camera_id']} - {cam.get('location', 'Unknown Location')}"
    # Lower threshold from 0.7 to 0.55 to prevent false matches between different people
    reid_manager = ReIDManager(db_config=None, similarity_threshold=0.55, camera_location=camera_location, debug_level=DEBUG_LEVEL)

    # Alias mapping for nicer display labels
    person_alias = {}
    person_names = {}  # Cache for person names
    
    def alias_for(pid):
        if pid is None:
            return '?'
        if pid not in person_alias:
            # Handle string IDs (like "M1", "M2") and integer IDs (1, 2, 3)
            if isinstance(pid, str):
                person_alias[pid] = pid  # Memory IDs already formatted as "M1"
            else:
                person_alias[pid] = f"P{pid:03d}"  # Database IDs as "P001", "P002"
        return person_alias[pid]
    
    def get_person_name(pid):
        """Get person name from database with caching"""
        if pid is None:
            return None
        if pid not in person_names:
            try:
                stats = reid_manager.db.get_person_stats(pid)
                if stats and stats[3]:  # stats[3] is the name field
                    person_names[pid] = stats[3]
                else:
                    person_names[pid] = None
            except Exception:
                person_names[pid] = None
        return person_names[pid]

    print("[INFO] CCTV pipeline started")
    print(f"[INFO] Debug level: {DEBUG_LEVEL} (0=Silent, 1=Summary, 2=Events, 3=Verbose)")
    print("=" * 80)
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'total_persons': 0,
        'total_violations': 0,
        'memory_persons': 0,
        'db_persons': 0,
        'alerts': 0,
        'warnings': 0
    }
    
    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]
        stats['total_frames'] += 1

        # Detection
        detections = detector.detect(frame)

        # per-class confidence filtering
        persons = [d for d in detections if d["class"] == "person" and d["confidence"] >= CONF_THRESHOLD_PERSON]
        masks = [d for d in detections if d["class"] == "mask" and d["confidence"] >= CONF_THRESHOLD_MASK]
        helmets = [d for d in detections if d["class"] == "helmet" and d["confidence"] >= CONF_THRESHOLD_HELMET]
        
        stats['total_persons'] += len(persons)
        
        # Debug output (verbose mode only)
        if DEBUG_LEVEL >= 3:
            try:
                print(f"[FRAME {frame_id}] Detections: {len(persons)}P {len(masks)}M {len(helmets)}H | "
                      f"Raw: {[(d['class'][:3], round(d['confidence'],2)) for d in detections]}")
            except Exception:
                pass
        
        # Draw all raw detections for debugging (before tracking)
        for mask_det in masks:
            x1, y1, x2, y2 = mask_det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan for masks
            cv2.putText(frame, f"MASK {mask_det['confidence']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        for helmet_det in helmets:
            x1, y1, x2, y2 = helmet_det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta for helmets
            cv2.putText(frame, f"HELMET {helmet_det['confidence']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Merge overlapping person boxes (simple NMS) to avoid duplicate detections
        def iou_box(a, b):
            xA = max(a[0], b[0])
            yA = max(a[1], b[1])
            xB = min(a[2], b[2])
            yB = min(a[3], b[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
            areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
            union = areaA + areaB - interArea
            return interArea / union if union > 0 else 0.0

        def nms_persons(dets, iou_thresh=0.5):
            out = []
            # sort by confidence desc
            dets_sorted = sorted(dets, key=lambda x: x.get('confidence', 0.0), reverse=True)
            for d in dets_sorted:
                keep = True
                for k in out:
                    if iou_box(d['bbox'], k['bbox']) > iou_thresh:
                        keep = False
                        break
                if keep:
                    out.append(d)
            return out

        if len(persons) > 1:
            persons = nms_persons(persons, iou_thresh=0.45)

        # face detection for mask association
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_boxes = [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in faces]

        # Tracking
        tracks = tracker.update(persons, frame)

        # helpers
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
            boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
            union = boxAArea + boxBArea - interArea
            if union == 0:
                return 0.0
            return interArea / union

        def head_overlap(person_bbox, attr_bbox, head_frac=HEAD_FRAC, iou_thresh=IOU_THRESHOLD):
            px1, py1, px2, py2 = person_bbox
            head_y2 = int(py1 + (py2 - py1) * head_frac)
            head_box = [px1, py1, px2, head_y2]
            return iou(head_box, attr_bbox) >= iou_thresh

        # Associate attributes with tracks (BEFORE ReID)
        # Priority: Helmet detection → Mask detection → ReID
        for t in tracks:
            track_id = t["track_id"]
            t["attributes"] = {"mask": False, "helmet": False}

            # check for mask (prefer face overlap)
            mask_present = False
            for mask_det in masks:
                matched_face = None
                for fb in face_boxes:
                    # face inside person bbox?
                    if fb[0] >= t["bbox"][0] and fb[2] <= t["bbox"][2] and fb[1] >= t["bbox"][1] and fb[3] <= t["bbox"][3]:
                        matched_face = fb
                        break

                if matched_face is not None:
                    if iou(matched_face, mask_det["bbox"]) >= IOU_THRESHOLD:
                        mask_present = True
                        break
                    else:
                        continue

                if head_overlap(t["bbox"], mask_det["bbox"]):
                    mask_present = True
                    break

            # check helmet
            helmet_present = False
            for helmet_det in helmets:
                if head_overlap(t["bbox"], helmet_det["bbox"]):
                    helmet_present = True
                    break

            # update smoothing history
            attribute_history[track_id]["mask"].append(1 if mask_present else 0)
            attribute_history[track_id]["helmet"].append(1 if helmet_present else 0)

            # apply smoothing thresholds
            if sum(attribute_history[track_id]["mask"]) >= ATTRIBUTE_REQUIRED_MASK:
                t["attributes"]["mask"] = True
            if sum(attribute_history[track_id]["helmet"]) >= ATTRIBUTE_REQUIRED_HELMET:
                t["attributes"]["helmet"] = True

            # Debug output (verbose mode only)
            if DEBUG_LEVEL >= 3:
                print(f"[Track {track_id}] M:{t['attributes']['mask']} ({sum(attribute_history[track_id]['mask'])}/{ATTRIBUTE_REQUIRED_MASK}) | "
                      f"H:{t['attributes']['helmet']} ({sum(attribute_history[track_id]['helmet'])}/{ATTRIBUTE_REQUIRED_HELMET})")

        # Re-identification: attach persistent person_id to tracks
        # Create mask/helmet dicts for ReID manager
        is_masked_dict = {t["track_id"]: t["attributes"]["mask"] for t in tracks}
        is_helmeted_dict = {t["track_id"]: t["attributes"]["helmet"] for t in tracks}
        
        try:
            tracks = reid_manager.update_tracking(tracks, frame, cam["camera_id"], is_masked_dict, is_helmeted_dict)
        except Exception as e:
            if DEBUG_LEVEL >= 2:
                print(f"[ReID] update_tracking failed: {e}")

        # Check for persons without names and send alerts
        if telegram is not None:
            for t in tracks:
                track_id = t["track_id"]
                person_id = t.get('person_id', None)
                
                # If person is registered but has no name assigned
                if person_id is not None:
                    person_name = get_person_name(person_id)
                    
                    # Alert if person has no name and we haven't alerted recently
                    if not person_name:
                        now = time.time()
                        if track_id not in unknown_person_alerted or (now - unknown_person_alerted[track_id]) >= 300:
                            unknown_person_alerted[track_id] = now
                            x1, y1, x2, y2 = t["bbox"]
                            telegram.send_alert(
                                frame, 
                                track_id, 
                                cam["camera_id"], 
                                "Unnamed Person",
                                f"Person ID {person_id} detected without name - please review and add name"
                            )
                            if DEBUG_LEVEL >= 2:
                                print(f"[UNNAMED] Person ID {person_id} (Track {track_id}) needs name assignment")

        # Motion and behavior
        motion_info = motion.update(tracks, frame_id)
        behavior_info = behavior.update(tracks, motion_info)
        
        # Track violation statistics
        for b in behavior_info:
            if b['decision'] == 'Alert':
                stats['alerts'] += 1
            elif b['decision'] == 'Warning':
                stats['warnings'] += 1

        # Draw and possibly send alerts
        for t, b in zip(tracks, behavior_info):
            track_id = t["track_id"]
            decision = b["decision"]
            reason = b["reason"]

            x1, y1, x2, y2 = t["bbox"]

            color = (255, 0, 0)
            if decision == "Warning":
                color = (0, 165, 255)
            elif decision == "Alert":
                color = (0, 0, 255)

            # Display stable person alias when available
            person_id = t.get('person_id', None)
            is_reidentified = t.get('is_reidentified', False)
            display_id = alias_for(person_id) if person_id is not None else f"T{track_id}"
            
            # Get person name if available
            person_name = get_person_name(person_id) if (person_id is not None and isinstance(person_id, int)) else None
            if person_name:
                display_label = f"{person_name} ({display_id}) | {decision}"
            else:
                display_label = f"{display_id} | {decision}"
            
            # Add re-identification indicator
            if is_reidentified:
                display_label += " [RE-ID]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                display_label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if telegram is not None:
                # Determine violation type from attributes
                attributes = t.get("attributes", {"mask": False, "helmet": False})
                violation_parts = []
                if attributes.get("helmet"):
                    violation_parts.append("HELMET")
                if attributes.get("mask"):
                    violation_parts.append("MASK")
                violation_type = "+".join(violation_parts) if violation_parts else reason
                
                if decision == "Alert":
                    if can_send(track_id, "Alert", cooldown=240):
                        telegram.send_alert(
                            frame, track_id, cam["camera_id"], decision, reason,
                            person_id=person_id,
                            is_reidentified=is_reidentified,
                            violation_type=violation_type
                        )
                        if DEBUG_LEVEL >= 2:
                            re_id_marker = " [RE-ID]" if is_reidentified else ""
                            print(f"[ALERT] {display_id}{re_id_marker} | {violation_type} | Camera: {cam['camera_id']}")

                elif decision == "Warning":
                    if can_send(track_id, "Warning", cooldown=240):
                        telegram.send_alert(
                            frame, track_id, cam["camera_id"], decision, reason,
                            person_id=person_id,
                            is_reidentified=is_reidentified,
                            violation_type=violation_type
                        )
                        if DEBUG_LEVEL >= 2:
                            re_id_marker = " [RE-ID]" if is_reidentified else ""
                            print(f"[WARNING] {display_id}{re_id_marker} | {violation_type} | Camera: {cam['camera_id']}")
        
        # Track person ID statistics
        current_memory = len([t for t in tracks if isinstance(t.get('person_id'), str) and t.get('person_id').startswith('M')])
        current_db = len([t for t in tracks if isinstance(t.get('person_id'), int)])
        stats['memory_persons'] = max(stats['memory_persons'], current_memory)
        stats['db_persons'] = max(stats['db_persons'], current_db)
        
        # Print summary every N frames (default 30 frames)
        if DEBUG_LEVEL >= 1 and frame_id % OUTPUT_EVERY_N_FRAMES == 0:
            avg_persons = stats['total_persons'] / stats['total_frames'] if stats['total_frames'] > 0 else 0
            print(f"\n{'='*80}")
            print(f"[SUMMARY] Frame: {frame_id} | Runtime: {int(frame_id/1)}s")
            print(f"  Current: {len(tracks)} tracked | {current_memory} in-memory | {current_db} in-database")
            print(f"  Averages: {avg_persons:.1f} persons/frame")
            print(f"  Violations: {stats['alerts']} alerts, {stats['warnings']} warnings")
            print(f"  Peak: {stats['memory_persons']} memory IDs, {stats['db_persons']} database IDs")
            print(f"{'='*80}\n")

        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Final summary
    if DEBUG_LEVEL >= 1:
        print(f"\n{'='*80}")
        print(f"[FINAL SUMMARY]")
        print(f"  Total frames processed: {stats['total_frames']}")
        print(f"  Total persons detected: {stats['total_persons']}")
        print(f"  Violations: {stats['alerts']} alerts, {stats['warnings']} warnings")
        print(f"  Peak tracking: {stats['memory_persons']} memory, {stats['db_persons']} database")
        print(f"{'='*80}\n")

    # Clean up
    try:
        reid_manager.close()
    except Exception:
        pass
    ingestor.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()