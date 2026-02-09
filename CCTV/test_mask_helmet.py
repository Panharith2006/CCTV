"""
Test trained mask/helmet detection model
Tests on single image, video, or live camera feed
Usage: python test_mask_helmet.py --source path/to/image.jpg
"""

from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
import time


def test_on_image(model, image_path, conf_threshold=0.25, save_result=True):
    """
    Test model on a single image
    
    Args:
        model: Loaded YOLO model
        image_path: Path to test image
        conf_threshold: Confidence threshold for detections
        save_result: Save annotated image
    """
    print(f"Testing on image: {image_path}")
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Process results
    for r in results:
        # Print detections
        boxes = r.boxes
        if len(boxes) > 0:
            print(f"\nDetections:")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  {label}: {conf:.2f} @ [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print("No detections")
        
        # Show annotated image
        annotated = r.plot()
        cv2.imshow('Detection Result - Press any key to continue', annotated)
        cv2.waitKey(0)
        
        # Save result
        if save_result:
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
            cv2.imwrite(output_path, annotated)
            print(f"Result saved to: {output_path}")
    
    cv2.destroyAllWindows()


def test_on_video(model, video_path, conf_threshold=0.25, save_result=True):
    """
    Test model on video file
    
    Args:
        model: Loaded YOLO model
        video_path: Path to video file
        conf_threshold: Confidence threshold
        save_result: Save output video
    """
    print(f"Testing on video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    writer = None
    if save_result:
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{Path(video_path).name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Annotate frame
        annotated = results[0].plot()
        
        # Count detections
        boxes = results[0].boxes
        mask_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == 'mask')
        helmet_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == 'helmet')
        
        # Add stats to frame
        stats_text = f"Frame: {frame_count}/{total_frames} | Masks: {mask_count} | Helmets: {helmet_count}"
        cv2.putText(annotated, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Video Detection - Press Q to stop', annotated)
        
        # Save frame
        if writer:
            writer.write(annotated)
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} fps)", end='\r')
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s (avg {avg_fps:.1f} fps)")


def test_on_camera(model, camera_id=0, conf_threshold=0.25):
    """
    Test model on live camera feed
    
    Args:
        model: Loaded YOLO model
        camera_id: Camera ID or RTSP URL
        conf_threshold: Confidence threshold
    """
    # Try to convert to int (webcam index) or keep as string (RTSP)
    try:
        camera_id = int(camera_id)
    except ValueError:
        pass
    
    print(f"Testing on camera: {camera_id}")
    print("Press 'q' to quit, 's' to save screenshot")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        
        # Count detections
        boxes = results[0].boxes
        mask_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == 'mask')
        helmet_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == 'helmet')
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add stats overlay
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Masks: {mask_count} | Helmets: {helmet_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Live Camera Detection - Q:quit S:screenshot', annotated)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"screenshot_{int(time.time())}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, annotated)
            print(f"Screenshot saved: {output_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {fps:.1f}")


def test_on_directory(model, images_dir, conf_threshold=0.25, save_results=True):
    """
    Test model on all images in a directory
    
    Args:
        model: Loaded YOLO model
        images_dir: Directory containing test images
        conf_threshold: Confidence threshold
        save_results: Save annotated images
    """
    print(f"Testing on directory: {images_dir}")
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    # Setup output directory
    if save_results:
        output_dir = "test_results/batch"
        os.makedirs(output_dir, exist_ok=True)
    
    total_detections = {'mask': 0, 'helmet': 0}
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {img_path.name}...", end=' ')
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        # Count detections
        boxes = results[0].boxes
        mask_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == 'mask')
        helmet_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == 'helmet')
        
        total_detections['mask'] += mask_count
        total_detections['helmet'] += helmet_count
        
        print(f"Masks: {mask_count}, Helmets: {helmet_count}")
        
        # Save result
        if save_results:
            annotated = results[0].plot()
            output_path = os.path.join(output_dir, f"result_{img_path.name}")
            cv2.imwrite(output_path, annotated)
    
    print(f"\n{'='*60}")
    print(f"BATCH TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Total mask detections: {total_detections['mask']}")
    print(f"Total helmet detections: {total_detections['helmet']}")
    if save_results:
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test mask/helmet detection model')
    parser.add_argument('--model', type=str, default='runs/mask_helmet/train_v1/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                       help='Test source: image path, video path, directory, camera ID, or RTSP URL')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results')
    parser.add_argument('--no-save', action='store_false', dest='save',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Train a model first: python train_mask_helmet.py")
        exit(1)
    
    model = YOLO(args.model)
    print(f"Model loaded successfully")
    print(f"Classes: {model.names}\n")
    
    # Determine source type and run appropriate test
    source = args.source
    
    if os.path.isfile(source):
        # Check if image or video
        ext = Path(source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_on_image(model, source, args.conf, args.save)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            test_on_video(model, source, args.conf, args.save)
        else:
            print(f"Unsupported file format: {ext}")
    
    elif os.path.isdir(source):
        # Directory of images
        test_on_directory(model, source, args.conf, args.save)
    
    else:
        # Assume camera (ID or RTSP)
        test_on_camera(model, source, args.conf)
