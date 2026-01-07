import cv2
import time
import os
from collections import defaultdict, deque

from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.two_stage_detector import TwoStageDetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer
from tracker.layer5_behavior import BehaviorDecider
from tracker.layer6_telegram import TelegramNotifier
from config.telegram_config import BOT_TOKEN, CHAT_ID


# ===================== TELEGRAM COOLDOWN =====================
last_telegram_sent = {}  # (track_id, decision) -> timestamp


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
    CONF_THRESHOLD_MASK = 0.25  # Lower for better mask recall
    CONF_THRESHOLD_HELMET = 0.4
    HEAD_FRAC = 0.45
    IOU_THRESHOLD = 0.1
    ATTRIBUTE_SMOOTHING = 5
    ATTRIBUTE_REQUIRED_MASK = 1  # Only need 1 detection in smoothing window
    ATTRIBUTE_REQUIRED_HELMET = 2
    MOTION_THRESHOLD = 150  # Higher = less sensitive to normal movement

    # Two-stage detector: person detector (COCO) + attribute detector (fine-tuned)
    detector = TwoStageDetector(person_model_path="yolov8n.pt", attr_model_path=model_path, conf_person=0.5, conf_attr=0.2)
    tracker = SortTracker()
    motion = MotionAnalyzer(frame_gaps=[1, 5, 10, 15, 20])

    behavior = BehaviorDecider(motion_threshold=MOTION_THRESHOLD, loitering_frames=10, warning_time=100, alert_time=200)

    # Create Telegram notifier only if credentials are set
    if BOT_TOKEN and CHAT_ID:
        telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    else:
        telegram = None

    # Attribute temporal smoothing storage
    attribute_history = defaultdict(lambda: {"mask": deque(maxlen=ATTRIBUTE_SMOOTHING), "helmet": deque(maxlen=ATTRIBUTE_SMOOTHING)})

    # Lightweight face detector for better mask association
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # Motion and behavior
        motion_info = motion.update(tracks, frame_id)
        for m in motion_info:
            max_motion = max(m["motion_gaps"].values()) if m["motion_gaps"] else 0
            print(f"[DEBUG] Track {m['track_id']} motion: {max_motion:.1f} (threshold: {MOTION_THRESHOLD})")
        behavior_info = behavior.update(tracks, motion_info)

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {decision}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if telegram is not None:
                if decision == "Alert":
                    if can_send(track_id, "Alert", cooldown=240):
                        telegram.send_alert(frame, track_id, cam["camera_id"], decision, reason)
                        print("[TELEGRAM] ALERT sent")

                elif decision == "Warning":
                    if can_send(track_id, "Warning", cooldown=240):
                        telegram.send_alert(frame, track_id, cam["camera_id"], decision, reason)
                        print("[TELEGRAM] WARNING sent")

        cv2.imshow("CCTV AI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
