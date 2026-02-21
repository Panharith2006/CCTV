import cv2
import time
import os
import sys
from collections import defaultdict, deque

from config.layer0_cameras import CAMERAS
from config.mysql_config import MYSQL_CONFIG
from ingest.layer1_frame_ingest import FrameIngestor
from detector.single_stage_detector import SingleStageDetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
from tracker.layer6_telegram import TelegramNotifier
from config.telegram_config import BOT_TOKEN, CHAT_ID
from tracker.violation_only_reid import ViolationOnlyReIDManager


# ===================== CONFIG FLAGS =====================
# Debug levels: 0=Silent, 1=Summary only, 2=Important events, 3=Verbose (every frame)
# 
# Recommended settings:
#   DEBUG_LEVEL = 0  → Production (no console output except errors)
#   DEBUG_LEVEL = 1  → Monitoring (summary every 30 frames, ~30 seconds)
#   DEBUG_LEVEL = 2  → Events (show violations, re-identifications, alerts)
#   DEBUG_LEVEL = 3  → Debug (show all detections, attributes, motion - VERY VERBOSE)
#
DEBUG_LEVEL = 2  # <-- CHANGED TO 2 TO SEE REID MATCHING DETAILS
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

    # Load trained model (person, mask, helmet)
    model_path = "best.pt"
    
    if not os.path.exists("best.pt"):
        print(f"[ERROR] Trained model 'best.pt' not found!")
        print(f"[ERROR] Please ensure best.pt (with person, mask, helmet classes) is in the project root")
        print(f"[ERROR] Or train a model using your training script")
        sys.exit(1)
    
    print(f"[INFO] Loading trained model: {model_path}")

    # Tunable thresholds (OPTIMIZED for violation detection accuracy)
    CONF_THRESHOLD_PERSON = 0.40    # LOWERED: Detect more people (safer for security)
    CONF_THRESHOLD_MASK = 0.50      # Balanced: Mask = WARNING level
    CONF_THRESHOLD_HELMET = 0.65    # HIGHER: Filter false helmet detections (model accuracy issues)
    HEAD_FRAC = 0.45
    IOU_THRESHOLD = 0.1
    ATTRIBUTE_SMOOTHING = 5
    ATTRIBUTE_REQUIRED_MASK = 2  # Need 2 detections in smoothing window for confirmation
    ATTRIBUTE_REQUIRED_HELMET = 3  # Need 3 detections in smoothing window for confirmation
    MOTION_THRESHOLD = 150  # Higher = less sensitive to normal movement
    LOITERING_WARNING_FRAMES = 360  # 6 minutes @ 1fps
    LOITERING_ALERT_FRAMES = 720  # 12 minutes @ 1fps

    # Single-stage detector: Detect person, then mask/helmet on head crops
    detector = SingleStageDetector(
        model_path=model_path,
        conf_person=CONF_THRESHOLD_PERSON, 
        conf_mask=CONF_THRESHOLD_MASK,
        conf_helmet=CONF_THRESHOLD_HELMET,
        head_fraction=HEAD_FRAC
    )
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

    # Violation-only ReID: Only tracks persons WITH mask/helmet violations
    # Normal persons → track_id only (no person_id, not saved)
    # Violators → person_id assigned, saved to database, re-identified on return
    camera_location = f"{cam['camera_id']} - {cam.get('location', 'Unknown Location')}"
    reid_manager = ViolationOnlyReIDManager(
        db_config=MYSQL_CONFIG,
        similarity_threshold=0.55,
        thumbnail_dir="thumbnails",
        camera_id=cam['camera_id'],
        camera_location=camera_location,
        debug_level=DEBUG_LEVEL
    )

    print("[INFO] CCTV pipeline started")
    print(f"[INFO] Debug level: {DEBUG_LEVEL} (0=Silent, 1=Summary, 2=Events, 3=Verbose)")
    print(f"[INFO] Detection strategy: Person → Head crop → Mask/Helmet")
    print(f"[INFO] Storage strategy: Violators ONLY (mask/helmet → database)")
    print("=" * 80)
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'total_persons': 0,
        'total_violations': 0,
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

        # Re-identification: Assign person_id ONLY to violators (mask/helmet)
        # Create mask/helmet dicts for ReID manager
        is_masked_dict = {t["track_id"]: t["attributes"]["mask"] for t in tracks}
        is_helmeted_dict = {t["track_id"]: t["attributes"]["helmet"] for t in tracks}
        
        try:
            tracks = reid_manager.update_tracks(frame, tracks, is_masked_dict, is_helmeted_dict)
        except Exception as e:
            if DEBUG_LEVEL >= 2:
                print(f"[ReID] update_tracks failed: {e}")

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

            color = (255, 0, 0)  # Blue for normal
            if decision == "Warning":
                color = (0, 165, 255)  # Orange for warning
            elif decision == "Alert":
                color = (0, 0, 255)  # Red for alert

            # Display person_id for violators, track_id for normal
            person_id = t.get('person_id', None)
            is_reidentified = t.get('is_reidentified', False)
            
            if person_id is not None:
                # Violator - show person ID
                display_id = f"P{person_id:03d}"
                if is_reidentified:
                    display_id += " [RE-ID]"
            else:
                # Normal person - just show track ID
                display_id = f"T{track_id}"
            
            display_label = f"{display_id} | {decision}"

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
                
                # Only send Telegram for ALERT priority (helmet, severe loitering)
                # WARNING priority (mask, moderate loitering) tracked but no Telegram sent
                if decision == "Alert":
                    if can_send(track_id, "Alert", cooldown=240):
                        telegram.send_alert(
                            frame, track_id, cam["camera_id"], decision, reason,
                            person_id=person_id,
                            is_reidentified=is_reidentified,
                            violation_type=violation_type
                        )
                        if DEBUG_LEVEL >= 2:
                            print(f"[ALERT] {display_id} | {violation_type} | Camera: {cam['camera_id']}")
                
                # WARNING alerts tracked in console/video but NOT sent to Telegram
                elif decision == "Warning" and DEBUG_LEVEL >= 2:
                    print(f"[WARNING] {display_id} | {violation_type} | Camera: {cam['camera_id']} (No Telegram)")
        
        # Track statistics
        current_violators = len([t for t in tracks if t.get('person_id') is not None])
        stats['total_violations'] = max(stats['total_violations'], current_violators)
        
        # Print summary every N frames (default 30 frames)
        if DEBUG_LEVEL >= 1 and frame_id % OUTPUT_EVERY_N_FRAMES == 0:
            avg_persons = stats['total_persons'] / stats['total_frames'] if stats['total_frames'] > 0 else 0
            reid_stats = reid_manager.get_statistics()
            print(f"\n{'='*80}")
            print(f"[SUMMARY] Frame: {frame_id} | Runtime: {int(frame_id/1)}s")
            print(f"  Tracked: {len(tracks)} persons | Violators: {current_violators}")
            print(f"  Averages: {avg_persons:.1f} persons/frame")
            print(f"  Violations: {stats['alerts']} alerts, {stats['warnings']} warnings")
            print(f"  Total violators seen: {reid_stats['total_violators']}, Normal: {reid_stats['total_normal']}")
            print(f"{'='*80}\n")

        cv2.imshow("CCTV AI - Violation Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Final summary
    if DEBUG_LEVEL >= 1:
        reid_stats = reid_manager.get_statistics()
        print(f"\n{'='*80}")
        print(f"[FINAL SUMMARY]")
        print(f"  Total frames processed: {stats['total_frames']}")
        print(f"  Total persons detected: {stats['total_persons']}")
        print(f"  Violations: {stats['alerts']} alerts, {stats['warnings']} warnings")
        print(f"  Violators tracked: {reid_stats['total_violators']}")
        print(f"  Normal persons ignored: {reid_stats['total_normal']}")
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