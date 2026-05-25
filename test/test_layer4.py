import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer4_motion_tracker import MotionAnalyzer

cam = CAMERAS[0]
ingestor = FrameIngestor(cam["camera_id"], cam["source"], sample_rate=3)
detector = YOLODetector(classes=["person"])
tracker = SortTracker()
motion = MotionAnalyzer(frame_gaps=[1,5,10,15,20])

for data in ingestor.read():
    frame = data["frame"]
    frame_id = data["frame_id"]

    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    motion_info = motion.update(tracks, frame_id)

    for m in motion_info:
        print(f"Track {m['track_id']} | Frame {m['frame_id']} | Gaps: {m['motion_gaps']}")
