"""
Manual Person Enrollment Tool
Usage: python -m tools.enroll_person
"""
import cv2
import sys
import os
import time
from detector.layer2_reid_extractor import ReIDExtractor
from database.reid_database import ReIDDatabase
from ultralytics import YOLO


def enroll_from_webcam(db_path="cctv_reid.db", name=None, captures=5, auto_interval=0.5):
    """Enroll a person by capturing multiple frames from webcam.

    This captures `captures` frames (default 5) and stores one person entry
    with multiple feature vectors (one per capture) for more robust matching.
    
    IMPORTANT: Uses YOLO person detection to match what main.py sees.
    """
    print("\n=== Manual Person Enrollment (multi-capture with detection) ===\n")
    print("Instructions:")
    print("  1. Stand 2-3 meters back from camera (full body visible)")
    print("  2. YOLO will detect you (green box appears)")
    print("  3. Press SPACE to start capture sequence")
    print("  4. Press ESC to cancel\n")

    # Initialize YOLO detector and ReID
    print("[INFO] Loading YOLO person detector...")
    yolo = YOLO("yolov8n.pt")
    reid = ReIDExtractor()
    db = ReIDDatabase(db_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return False

    # Determine number of captures
    if name is None:
        name_input = input("Enter person's name (or press Enter to skip): ").strip()
        person_name = name_input if name_input else None
    else:
        person_name = name

    try:
        n_input = input(f"Number of captures (default {captures}): ").strip()
        captures = int(n_input) if n_input else captures
    except Exception:
        captures = captures

    enrolled = False

    print("\n[INFO] Stand back and ensure full body is visible...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO person detection
        results = yolo(frame, classes=[0], verbose=False)  # class 0 = person
        display_frame = frame.copy()
        
        detected_person = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the largest person detection
            boxes = results[0].boxes
            areas = [(box.xyxy[0], (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])) 
                     for box in boxes]
            if areas:
                largest_box, _ = max(areas, key=lambda x: x[1])
                x1, y1, x2, y2 = map(int, largest_box.tolist())
                detected_person = [x1, y1, x2, y2]
                
                # Draw detection box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "Person Detected - Press SPACE", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display overlay
        cv2.putText(display_frame, f"Capture {captures} frames | ESC to cancel", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if detected_person is None:
            cv2.putText(display_frame, "No person detected - stand back!", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Enroll Person", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\n[INFO] Enrollment cancelled")
            break
        elif key == 32:  # SPACE
            if detected_person is None:
                print("[WARN] No person detected! Stand back and try again.")
                continue
                
            print(f"\n[INFO] Capturing {captures} frames with detection...")
            captured_data = []
            
            # Capture current frame
            captured_data.append((frame.copy(), detected_person))
            
            # Capture additional frames
            for i in range(captures - 1):
                time.sleep(auto_interval)
                ret2, f2 = cap.read()
                if not ret2:
                    break
                    
                # Detect person in this frame too
                results2 = yolo(f2, classes=[0], verbose=False)
                if len(results2) > 0 and len(results2[0].boxes) > 0:
                    boxes2 = results2[0].boxes
                    areas2 = [(box.xyxy[0], (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])) 
                             for box in boxes2]
                    if areas2:
                        largest_box2, _ = max(areas2, key=lambda x: x[1])
                        x1_2, y1_2, x2_2, y2_2 = map(int, largest_box2.tolist())
                        detected_person2 = [x1_2, y1_2, x2_2, y2_2]
                        captured_data.append((f2.copy(), detected_person2))

            # Extract features for each capture (original + horizontal flip)
            feature_list = []
            for idx, (fr, bbox) in enumerate(captured_data):
                # Original
                feats = reid.extract_features(fr, bbox)
                if feats is None:
                    print(f"[WARN] Capture {idx+1}: failed to extract features")
                    continue
                feature_list.append((feats, fr, bbox))
                
                # Flip augmentation
                fr_flip = cv2.flip(fr, 1)  # horizontal flip
                x1, y1, x2, y2 = bbox
                w = fr.shape[1]
                bbox_flip = [w - x2, y1, w - x1, y2]  # mirror bbox
                feats_flip = reid.extract_features(fr_flip, bbox_flip)
                if feats_flip is not None:
                    feature_list.append((feats_flip, fr_flip, bbox_flip))

            if len(feature_list) == 0:
                print("[ERROR] No valid captures were obtained")
                continue

            # Save thumbnail from first valid capture (cropped to person)
            thumbnail_dir = "thumbnails"
            os.makedirs(thumbnail_dir, exist_ok=True)
            first_frame, first_bbox = feature_list[0][1], feature_list[0][2]
            x1, y1, x2, y2 = first_bbox
            person_crop = first_frame[y1:y2, x1:x2]
            thumbnail_path = os.path.join(thumbnail_dir, f"enroll_{person_name or 'unknown'}.jpg")
            cv2.imwrite(thumbnail_path, person_crop)

            # Add first feature as new person, then add remaining features as updates
            first_feat = feature_list[0][0]
            person_id = db.add_person(first_feat, camera_id="manual", thumbnail_path=thumbnail_path, name=person_name)

            added = 1
            for feats, _, _ in feature_list[1:]:
                db.update_person(person_id, feats, camera_id="manual")
                added += 1

            print(f"\n✓ Successfully enrolled!")
            print(f"  Person ID: {person_id}")
            print(f"  Name: {person_name or '(unnamed)'}")
            print(f"  Thumbnail: {thumbnail_path}")
            print(f"  Feature vectors stored: {added}")
            print(f"  Detection-based enrollment (matches main.py)\n")

            enrolled = True
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()

    if enrolled:
        print("[INFO] Enrollment complete. You can now run the main system.")

    return enrolled


def enroll_from_image(image_path, db_path="cctv_reid.db", name=None):
    """Enroll a person from an image file"""
    print(f"\n=== Enrolling from image: {image_path} ===\n")
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return False
    
    # Initialize
    reid = ReIDExtractor()
    db = ReIDDatabase(db_path)

    # Try to detect person using YOLO so we crop like main pipeline
    try:
        yolo = YOLO("yolov8n.pt")
        results = yolo(frame, classes=[0], verbose=False)
        bbox = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            areas = [(box.xyxy[0], (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])) for box in boxes]
            if areas:
                largest_box, _ = max(areas, key=lambda x: x[1])
                x1, y1, x2, y2 = map(int, largest_box.tolist())
                bbox = [x1, y1, x2, y2]
                print(f"[INFO] Detected person in image, using bbox={bbox}")
        if bbox is None:
            print("[WARN] No person detected in image; falling back to full image crop")
            bbox = [0, 0, frame.shape[1], frame.shape[0]]

    except Exception:
        print("[WARN] YOLO detection failed; using full image crop")
        bbox = [0, 0, frame.shape[1], frame.shape[0]]

    print("[INFO] Extracting features...")
    features = reid.extract_features(frame, bbox)
    
    if features is None:
        print("[ERROR] Failed to extract features")
        return False
    
    # Get name from user if not provided
    if name is None:
        name_input = input("\nEnter person's name (or press Enter to skip): ").strip()
        person_name = name_input if name_input else None
    else:
        person_name = name
    
    # Save thumbnail (crop to person)
    thumbnail_dir = "thumbnails"
    os.makedirs(thumbnail_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, bbox)
    person_crop = frame[y1:y2, x1:x2]
    thumbnail_path = os.path.join(thumbnail_dir, f"enroll_{person_name or 'unknown'}.jpg")
    cv2.imwrite(thumbnail_path, person_crop)
    
    # Add to database
    person_id = db.add_person(features, camera_id="manual", 
                             thumbnail_path=thumbnail_path, name=person_name)
    
    print(f"\n✓ Successfully enrolled!")
    print(f"  Person ID: {person_id}")
    print(f"  Name: {person_name or '(unnamed)'}")
    print(f"  Thumbnail: {thumbnail_path}")
    print(f"  Feature vector shape: {features.shape}\n")
    
    db.close()
    return True


def main():
    """Main enrollment interface"""
    print("\n" + "="*50)
    print(" "*10 + "Person Enrollment Tool")
    print("="*50)
    
    print("\nOptions:")
    print("  1. Enroll from webcam")
    print("  2. Enroll from image file")
    print("  3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        name = input("Enter person's name (e.g., 'Rith'): ").strip() or None
        enroll_from_webcam(name=name)
    
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        name = input("Enter person's name (e.g., 'Rith'): ").strip() or None
        enroll_from_image(image_path, name=name)
    
    elif choice == "3":
        print("\n[INFO] Exiting...")
        return
    
    else:
        print("\n[ERROR] Invalid option")


if __name__ == "__main__":
    main()
