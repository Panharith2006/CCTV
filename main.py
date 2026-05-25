import cv2
import time

from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
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

    detector = YOLODetector(classes=["person"])
    tracker = SortTracker()
    motion = MotionAnalyzer(frame_gaps=[1, 5, 10, 15, 20])

    behavior = BehaviorDecider(
        warning_time=100,   # warning at 100s
        alert_time=200      # alert at 200s
    )

    telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)

    print("[INFO] CCTV pipeline started")

    for data in ingestor.read():
        frame = data["frame"]
        frame_id = data["frame_id"]

        detections = detector.detect(frame)
        persons = [d for d in detections if d["class"] == "person"]

        tracks = tracker.update(persons, frame)

        motion_info = motion.update(tracks, frame_id)
        behavior_info = behavior.update(tracks, motion_info)

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

            # -------- TELEGRAM LOGIC --------
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
