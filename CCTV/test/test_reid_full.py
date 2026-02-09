import cv2
from config.layer0_cameras import CAMERAS
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from tracker.layer3_sort_tracker import SortTracker
from tracker.layer3_reid_manager import ReIDManager


def main():
    print("=== Full ReID Pipeline Test ===\n")
    
    # Initialize
    cam = CAMERAS[0]
    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )
    
    detector = YOLODetector(classes=["person"], conf_threshold=0.5)
    tracker = SortTracker()
    reid_manager = ReIDManager(db_path="test_reid.db", similarity_threshold=0.7)
    
    print("\n[INFO] System ready!")
    print("[INFO] Walk in and out of frame to test re-identification")
    print("[INFO] Each unique person should get a consistent Person ID\n")
    
    frame_count = 0
    
    for data in ingestor.read():
        frame = data["frame"]
        frame_count += 1
        
        # Detect persons
        detections = detector.detect(frame)
        persons = [d for d in detections if d["class"] == "person"]
        
        # Track
        tracks = tracker.update(persons, frame)
        
        # ReID
        tracks = reid_manager.update_tracking(tracks, frame, cam["camera_id"])
        
        # Visualize
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            person_id = track.get('person_id', None)
            
            # Color based on person_id
            if person_id is not None:
                color_idx = person_id % 10
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), 
                         (255,0,255), (0,255,255), (128,0,128), (255,128,0),
                         (128,255,0), (0,128,255)]
                color = colors[color_idx]
            else:
                color = (128, 128, 128)
                person_id = '?'
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Person-{person_id} | Track-{track_id}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info
        cv2.putText(frame, f"Frame: {frame_count} | Persons: {len(tracks)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "ESC to exit", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Full ReID Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    reid_manager.close()
    ingestor.release()
    cv2.destroyAllWindows()
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
