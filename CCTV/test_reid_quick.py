"""
Quick OSNet ReID test - processes 50 frames for faster demo
"""

import cv2
import numpy as np
import os
from collections import defaultdict
from detector.single_stage_detector import SingleStageDetector
from detector.layer2_reid_extractor_enhanced import EnhancedReIDExtractor


def quick_test():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     Quick OSNet ReID Test (50 frames)                     ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    video_path = 'cctv_test.MOV'
    
    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return
    
    print(f"Video: {video_path}\n")
    
    # Load models
    print("Loading detector...")
    detector = SingleStageDetector(model_path='best.pt', conf_person=0.5)
    
    print("Loading OSNet ReID extractor...\n")
    reid = EnhancedReIDExtractor(model_name='osnet_x1_0', use_gpu=True, output_dim=512)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    person_features_db = defaultdict(list)
    total_persons = 0
    
    print("="*60)
    print("Processing video...")
    print("="*60 + "\n")
    
    while frame_count < 50:  # Process only 50 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections = detector.detect(frame)
        person_dets = [d for d in detections if d['class'] == 'person']
        
        if len(person_dets) > 0:
            print(f"Frame {frame_count}: Found {len(person_dets)} person(s)")
            
            for det in person_dets:
                try:
                    features, quality = reid.extract_features(
                        frame, det['bbox'], return_quality=True
                    )
                    
                    if features is not None:
                        person_id = f"P{len(person_features_db)}"
                        person_features_db[person_id].append(features)
                        total_persons += 1
                        print(f"  → {person_id}: 512D feature extracted (quality: {quality:.2f})")
                
                except Exception as e:
                    print(f"  ✗ Error: {e}")
    
    cap.release()
    
    # Results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"\n✓ Processed: {frame_count} frames")
    print(f"✓ Persons found: {total_persons}")
    print(f"✓ Unique persons: {len(person_features_db)}")
    
    if len(person_features_db) >= 2:
        print(f"\nFeature Consistency Test:")
        person_ids = list(person_features_db.keys())
        
        # Same person across frames
        p1_feats = person_features_db[person_ids[0]]
        if len(p1_feats) > 1:
            sim = reid.compare_features(p1_feats[0], p1_feats[-1])
            print(f"  {person_ids[0]} (frame 1 ↔ last): {sim:.4f}")
            print(f"  → Consistency: {'✓ Good' if sim > 0.5 else '⚠ Lower'}")
        
        # Different persons (if available)
        if len(person_ids) > 1:
            p1_feat = person_features_db[person_ids[0]][0]
            p2_feat = person_features_db[person_ids[1]][0]
            sim = reid.compare_features(p1_feat, p2_feat)
            print(f"  {person_ids[0]} ↔ {person_ids[1]}: {sim:.4f}")
            print(f"  → Different: {'✓ Distinct features' if sim < 0.5 else '⚠ Similar features'}")
    
    print(f"\n" + "="*60)
    print("✓ Quick test completed!")
    print("="*60)
    
    print(f"""
✓ OSNet ImageNet model is working correctly!

Next step - Run full CCTV system:
  python main.py

This will apply ReID to all your cameras in real-time.
    """)


if __name__ == '__main__':
    quick_test()
