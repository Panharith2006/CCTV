"""
Interactive tool to collect training images for mask/helmet detection.

Usage:
    python collect_training_data.py

Controls:
    SPACE - Capture image
    Q/ESC - Quit
    
You'll need to label the images afterward using a tool like LabelImg or Roboflow.
"""
import cv2
import os
from datetime import datetime


def main():
    # Output directory
    output_dir = "dataset/raw_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("TRAINING DATA COLLECTION TOOL")
    print("="*60)
    print(f"Images will be saved to: {output_dir}")
    print("\nInstructions:")
    print("  1. Press SPACE to capture image")
    print("  2. Collect images with:")
    print("     - People WITHOUT masks (normal)")
    print("     - People WITH masks (abnormal)")
    print("     - People WITH helmets (abnormal)")
    print("     - Different angles, lighting, distances")
    print("  3. Press Q or ESC to quit")
    print("\nTip: Collect at least 100-200 images for good results")
    print("="*60)
    
    cap = cv2.VideoCapture(0)  # Webcam
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            break
        
        # Display instructions on frame
        display = frame.copy()
        cv2.putText(display, f"Images captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=Capture | Q=Quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Collect Training Data", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capture image
        if key == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"img_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            count += 1
            print(f"[{count}] Saved: {filename}")
        
        # Quit
        elif key == ord('q') or key == 27:  # Q or ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"Collection complete! {count} images saved to {output_dir}")
    print("\nNext steps:")
    print("  1. Label your images using:")
    print("     - Roboflow: https://roboflow.com (recommended)")
    print("     - LabelImg: https://github.com/heartexlabs/labelImg")
    print("  2. Export in YOLO format")
    print("  3. Copy to dataset/mask_dataset/")
    print("  4. Run: python train_mask.py --epochs 50 --imgsz 640")
    print("="*60)


if __name__ == "__main__":
    main()
