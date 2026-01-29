import cv2
import numpy as np
from detector.layer2_reid_extractor import ReIDExtractor
from ingest.layer1_frame_ingest import FrameIngestor
from detector.layer2_yolo_detector import YOLODetector
from config.layer0_cameras import CAMERAS


def main():
    print("=== Testing ReID Feature Extraction ===\n")
    
    # Initialize components
    cam = CAMERAS[0]
    ingestor = FrameIngestor(
        camera_id=cam["camera_id"],
        source=cam["source"],
        sample_rate=3
    )
    
    detector = YOLODetector(classes=["person"], conf_threshold=0.5)
    reid_extractor = ReIDExtractor()
    
    # Store first person's features
    reference_features = None
    reference_bbox = None
    
    print("\n[INFO] Detecting first person as reference...")
    
    for data in ingestor.read():
        frame = data["frame"]
        detections = detector.detect(frame)
        
        # Filter person detections
        persons = [d for d in detections if d["class"] == "person"]
        
        if len(persons) == 0:
            cv2.putText(frame, "No person detected - move into view", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("ReID Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        # Use first detection as reference
        if reference_features is None:
            person = persons[0]
            reference_bbox = person["bbox"]
            reference_features = reid_extractor.extract_features(frame, reference_bbox)
            
            if reference_features is not None:
                print(f"\n✓ Reference person captured!")
                print(f"  Feature vector shape: {reference_features.shape}")
                print(f"  Feature vector (first 10 dims): {reference_features[:10]}")
                print(f"\n[INFO] Now detecting and comparing all persons...\n")
        
        # Compare all detected persons with reference
        for i, person in enumerate(persons):
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Extract features
            features = reid_extractor.extract_features(frame, bbox)
            
            if features is not None and reference_features is not None:
                # Compute similarity
                similarity = reid_extractor.compare_features(reference_features, features)
                
                # Determine if same person (threshold = 0.7)
                is_same = similarity > 0.7
                color = (0, 255, 0) if is_same else (0, 0, 255)
                label = f"Person {i+1} | Sim: {similarity:.3f} | {'MATCH' if is_same else 'DIFFERENT'}"
                
                print(f"  Person {i+1}: Similarity = {similarity:.3f} ({'SAME' if is_same else 'DIFFERENT'})")
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display reference bbox
        if reference_bbox is not None:
            rx1, ry1, rx2, ry2 = reference_bbox
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
            cv2.putText(frame, "REFERENCE", (rx1, ry1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(frame, "ESC to exit | Green=Match, Red=Different, Blue=Reference", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("ReID Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    ingestor.release()
    cv2.destroyAllWindows()
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
