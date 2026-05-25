import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector

def main():
    cam = CAMERAS[0]

    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )

    detector = YOLODetector(classes=["person", "mask", "helmet"])

    for data in ingestor.read():
        frame = data["frame"]
        detections = detector.detect(frame)

        print(f"[Layer2] Detections: {len(detections)}")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Layer 2 - YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    ingestor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
