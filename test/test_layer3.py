import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker

def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person"])
    tracker = SortTracker()

    for data in ingestor.read():
        frame = data["frame"]

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Layer 3 - SORT Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
