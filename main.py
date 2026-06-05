import cv2
import time
import os
from collections import defaultdict, deque

from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer6_telegram import TelegramNotifier
from config.telegram_config import BOT_TOKEN, CHAT_ID
from tracker.layer3_reid_manager import ReIDManager


# ===================== TELEGRAM COOLDOWN =====================
unknown_person_alerted = {}  # track_id -> timestamp (to track unknown person alerts)

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
    repo_root = os.path.dirname(os.path.abspath(__file__))

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    # Single YOLO model for person, mask, and helmet detection.
    # best.pt is the only model file used by the runtime detector.
    model_path = os.path.join(repo_root, "best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing YOLO model: {model_path}")

    print(f"[INFO] Loading YOLO model: {model_path}")

    # Tunable thresholds
    # Align person filter with detector's person confidence (was filtering out detections)
    CONF_THRESHOLD_PERSON = 0.5
    CONF_THRESHOLD_MASK = 0.25  # Lower for better mask recall
    CONF_THRESHOLD_HELMET = 0.4
    HEAD_FRAC = 0.45
    IOU_THRESHOLD = 0.1
    ATTRIBUTE_SMOOTHING = 30
    ATTRIBUTE_REQUIRED_MASK = 3  # Only need 3 detections in a 30-frame window
    ATTRIBUTE_REQUIRED_HELMET = 5

    # Single-model detector using best.pt only.
    detector = YOLODetector(model_path=model_path, classes=["person", "mask", "helmet"], conf_threshold=0.2)
    tracker = SortTracker()

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
    REID_REQUIRED_CONFIRM = 50  # frames of consistent PPE before ReID/enrollment (approx 5.0 seconds)
    reid_manager = ReIDManager(db_config=None, similarity_threshold=0.7, required_confirm_frames=REID_REQUIRED_CONFIRM)

    # Alias mapping for nicer display labels
    person_alias = {}
    person_names = {}  # Cache for person names
    
    def alias_for(pid):
        if pid is None:
            return '?'
        if pid not in person_alias:
            person_alias[pid] = f"P{pid:03d}"
        return person_alias[pid]
    
    def get_person_name(pid):
        """Get person name from database with caching"""
        if pid is None:
            return None
        if pid not in person_names:
            try:
                stats = reid_manager.db.get_person_stats(pid)
                if stats and stats.get('name'):
                    person_names[pid] = stats.get('name')
                else:
                    person_names[pid] = None
            except Exception:
                person_names[pid] = None
        return person_names[pid]

    print("[INFO] CCTV pipeline started")

    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]

        # Detection
        detections = detector.detect(frame)

        # Debug: list raw detections from detector
        try:
            print(f"[DEBUG] Raw detections: {[ (d['class'], round(d['confidence'],2), d['bbox']) for d in detections ]}")
        except Exception:
            pass

        # per-class confidence filtering
        persons = [d for d in detections if d["class"] == "person" and d["confidence"] >= CONF_THRESHOLD_PERSON]
        masks = [d for d in detections if d["class"] == "mask" and d["confidence"] >= CONF_THRESHOLD_MASK]
        helmets = [d for d in detections if d["class"] == "helmet" and d["confidence"] >= CONF_THRESHOLD_HELMET]
        
        print(f"[DEBUG] Frame {frame_id}: {len(persons)} persons, {len(masks)} masks, {len(helmets)} helmets")
        
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

        # Draw raw person detections immediately so boxes are visible before SORT confirms tracks.
        for person_det in persons:
            x1, y1, x2, y2 = person_det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"PERSON {person_det['confidence']:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

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

        # Associate attributes with tracks
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

            # Debug output
            print(f"[DEBUG] Track {track_id}: mask={t['attributes']['mask']} (hist: {list(attribute_history[track_id]['mask'])}), helmet={t['attributes']['helmet']} (hist: {list(attribute_history[track_id]['helmet'])})")

        # Re-identification only after PPE attributes are known.
        try:
            tracks = reid_manager.update_tracking(tracks, frame, cam["camera_id"], require_ppe=True)
        except Exception as e:
            print(f"[ReID] update_tracking failed: {e}")

        # Send alerts for detected persons with PPE
        if telegram is not None:
            for t in tracks:
                track_id = t["track_id"]
                person_id = t.get('person_id', None)
                
                # If person has been confirmed (assigned an ID)
                if person_id is not None:
                    person_name = get_person_name(person_id)
                    display_name = person_name if person_name else "Unknown"
                    
                    now = time.time()
                    if track_id not in unknown_person_alerted or (now - unknown_person_alerted[track_id]) >= 300:
                        unknown_person_alerted[track_id] = now
                        x1, y1, x2, y2 = t["bbox"]
                        
                        # Extract the cropped image of the person
                        h, w = frame.shape[:2]
                        c_x1, c_y1 = max(0, x1), max(0, y1)
                        c_x2, c_y2 = min(w, x2), min(h, y2)
                        person_crop = frame[c_y1:c_y2, c_x1:c_x2]
                        
                        if person_crop.size > 0:
                            attributes = t.get("attributes", {})
                            has_mask = attributes.get("mask", False)
                            has_helmet = attributes.get("helmet", False)
                            
                            if has_mask:
                                ppe_text = "a mask"
                            elif has_helmet:
                                ppe_text = "a helmet"
                            else:
                                ppe_text = "PPE"
                                
                            msg_text = f"Person '{display_name}' (ID {person_id}) detected wearing {ppe_text}. Please review."
                            
                            # Send the full frame first
                            telegram.send_alert(
                                frame, 
                                track_id, 
                                cam["camera_id"], 
                                "Abnormal Detection - Full Scene",
                                msg_text
                            )
                            # Send the cropped image second
                            telegram.send_alert(
                                person_crop, 
                                track_id, 
                                cam["camera_id"], 
                                "Abnormal Detection - Cropped",
                                ""
                            )
                            print(f"[TELEGRAM] Alert sent for Person '{display_name}' (ID {person_id}), track {track_id}")

        # Draw tracked people and send alerts for named/unnamed registration only.
        for t in tracks:
            track_id = t["track_id"]

            x1, y1, x2, y2 = t["bbox"]

            color = (255, 0, 0)
            if t.get("attributes", {}).get("mask") or t.get("attributes", {}).get("helmet"):
                color = (0, 255, 0)

            # Display stable person alias when available
            # If ReID hasn't assigned a person_id yet we don't show the temporary "T" placeholder.
            # YOLO already provides class labels (person/mask/helmet); tracking/ReID add identity overlays.
            person_id = t.get('person_id', None)
            display_id = alias_for(person_id) if person_id is not None else None

            # Get person name if available
            person_name = get_person_name(person_id) if person_id is not None else None
            if person_name:
                # Show the registered name and short alias
                display_label = f"{person_name} ({display_id})"
            else:
                # If unknown person (no ReID yet), don't draw a 'T' placeholder; leave label empty
                display_label = ""

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if display_label:
                cv2.putText(
                    frame,
                    display_label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up
    try:
        reid_manager.close()
    except Exception:
        pass
    ingestor.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
