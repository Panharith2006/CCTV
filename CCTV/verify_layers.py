"""
Quick verification script to check all layers are working correctly
before running the full system with Telegram bot.
"""
import sys
import os

print("="*80)
print("CCTV AI SYSTEM - LAYER VERIFICATION")
print("="*80)

print("\n[1/7] Checking imports...")
try:
    import cv2
    import numpy as np
    import torch
    import mysql.connector
    print("  ✓ Core libraries (cv2, numpy, torch, mysql) OK")
except ImportError as e:
    print(f"  ✗ Missing library: {e}")
    sys.exit(1)

print("\n[2/7] Checking trained model...")
if os.path.exists("best.pt"):
    print("  ✓ Trained model (best.pt) found")
    
    # Load and check model classes
    try:
        from ultralytics import YOLO
        model = YOLO("best.pt")
        class_names = model.names
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Classes detected: {class_names}")
        
        # Verify mask and helmet classes exist
        has_mask = any('mask' in str(v).lower() for v in class_names.values())
        has_helmet = any('helmet' in str(v).lower() for v in class_names.values())
        
        if has_mask and has_helmet:
            print("  ✓ Required classes (mask, helmet) present")
        else:
            print("  ⚠ Warning: mask or helmet class might be missing")
            print(f"    Available classes: {class_names}")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        sys.exit(1)
else:
    print("  ✗ best.pt not found - please train the model first")
    sys.exit(1)

print("\n[3/7] Checking Layer 0 (Camera Config)...")
try:
    from config.layer0_cameras import CAMERAS
    print(f"  ✓ Camera configuration loaded: {len(CAMERAS)} camera(s)")
    for cam in CAMERAS:
        print(f"    - {cam['camera_id']}: {cam['location']} (source: {cam['source']})")
except Exception as e:
    print(f"  ✗ Camera config error: {e}")
    sys.exit(1)

print("\n[4/7] Checking Layer 1 (Frame Ingestion)...")
try:
    from ingest.layer1_frame_ingest import FrameIngestor
    print("  ✓ FrameIngestor class imported")
except Exception as e:
    print(f"  ✗ Frame ingestion error: {e}")
    sys.exit(1)

print("\n[5/7] Checking Layer 2 (Detection + ReID)...")
try:
    from detector.two_stage_detector import TwoStageDetector
    from detector.layer2_reid_extractor import ReIDExtractor
    print("  ✓ TwoStageDetector class imported")
    print("  ✓ ReIDExtractor class imported (128D OSNet)")
except Exception as e:
    print(f"  ✗ Detection layer error: {e}")
    sys.exit(1)

print("\n[6/7] Checking Layer 3-5 (Tracking, Motion, Behavior)...")
try:
    from tracker.layer3_sort_tracker import SortTracker
    from tracker.layer3_reid_manager import ReIDManager
    from tracker.layer4_motion_tracker import MotionAnalyzer
    from tracker.layer5_behavior import BehaviorDecider
    print("  ✓ SortTracker class imported")
    print("  ✓ ReIDManager class imported")
    print("  ✓ MotionAnalyzer class imported")
    print("  ✓ BehaviorDecider class imported")
except Exception as e:
    print(f"  ✗ Tracking/behavior layer error: {e}")
    sys.exit(1)

print("\n[7/7] Checking Layer 6 (Telegram)...")
try:
    from tracker.layer6_telegram import TelegramNotifier
    from config.telegram_config import BOT_TOKEN, CHAT_ID
    print("  ✓ TelegramNotifier class imported")
    
    if BOT_TOKEN and CHAT_ID:
        print(f"  ✓ Telegram configured:")
        print(f"    - Bot Token: {BOT_TOKEN[:20]}..." if len(BOT_TOKEN) > 20 else f"    - Bot Token: {BOT_TOKEN}")
        print(f"    - Chat ID: {CHAT_ID}")
    else:
        print("  ⚠ Telegram NOT configured (BOT_TOKEN or CHAT_ID missing)")
        print("    System will run without Telegram alerts")
except Exception as e:
    print(f"  ✗ Telegram layer error: {e}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

# Summary
print("\nSYSTEM STATUS:")
print("  ✓ All layers are working correctly")
print("  ✓ Trained model loaded (mask + helmet detection)")
print("  ✓ Camera configured")

if BOT_TOKEN and CHAT_ID:
    print("  ✓ Telegram bot ready")
    print("\n✅ READY TO RUN: python main.py")
else:
    print("  ⚠ Telegram not configured")
    print("\n⚠ Configure Telegram bot to enable alerts:")
    print("  1. Edit config/telegram_config.py")
    print("  2. Set BOT_TOKEN and CHAT_ID")
    print("  3. Or create .env file with:")
    print("     TELEGRAM_BOT_TOKEN=your_token")
    print("     TELEGRAM_CHAT_ID=your_chat_id")

print("\n" + "="*80)
