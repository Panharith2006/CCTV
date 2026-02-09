"""
Extract frames from CCTV video footage for mask/helmet dataset creation
Usage: python scripts/extract_frames.py --video path/to/video.mp4 --output data/raw_images
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=30, max_frames=None):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30 = 1 per second for 30fps)
        max_frames: Maximum number of frames to extract (None = all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video Info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting every {frame_interval} frames")
    print(f"  Expected output: ~{total_frames // frame_interval} frames\n")
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at interval
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            # Progress indicator
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames...", end='\r')
            
            # Stop if max_frames reached
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"\nComplete! Extracted {saved_count} frames to {output_dir}")


def extract_from_camera(camera_id, output_dir, duration_seconds=60, frame_interval=30):
    """
    Extract frames from live camera feed
    
    Args:
        camera_id: Camera ID or RTSP URL
        output_dir: Directory to save frames
        duration_seconds: How long to capture (seconds)
        frame_interval: Extract every Nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera: {camera_id}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int((fps * duration_seconds) // frame_interval)
    
    print(f"Capturing from camera: {camera_id}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Expected frames: ~{max_frames}\n")
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        # Display preview
        cv2.imshow('Capturing - Press Q to stop', frame)
        
        # Save frame at interval
        if frame_count % frame_interval == 0:
            filename = f"cam_frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f"Captured {saved_count}/{max_frames} frames", end='\r')
        
        frame_count += 1
        
        # Allow early exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nComplete! Captured {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video for dataset creation')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', type=str, help='Camera ID or RTSP URL')
    parser.add_argument('--output', type=str, default='data/raw_images', 
                       help='Output directory (default: data/raw_images)')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Frame interval (default: 30 = 1 per second at 30fps)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to extract (default: all)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for camera capture in seconds (default: 60)')
    
    args = parser.parse_args()
    
    if args.video:
        extract_frames(args.video, args.output, args.interval, args.max_frames)
    elif args.camera:
        # Convert to int if numeric, else use as RTSP URL
        try:
            camera_id = int(args.camera)
        except ValueError:
            camera_id = args.camera
        extract_from_camera(camera_id, args.output, args.duration, args.interval)
    else:
        print("Error: Please specify either --video or --camera")
        print("\nExamples:")
        print("  python scripts/extract_frames.py --video footage.mp4")
        print("  python scripts/extract_frames.py --camera 0")
        print("  python scripts/extract_frames.py --camera rtsp://admin:pass@192.168.1.100:554/stream")
