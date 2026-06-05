"""
CCTV System Health Check
Run this to validate all components are working correctly
"""
import cv2
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector.layer2_reid_extractor import ReIDExtractor
from database.reid_database import ReIDDatabase

print("=== CCTV System Health Check ===\n")

# 1. Camera Test
print("1. Camera Test...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"   ✓ Camera working: {frame.shape}")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("   ✗ Camera frame read failed")
    cap.release()
else:
    print("   ✗ Camera not opened (try different index or check connection)")

# 2. YOLO Detection Test
print("\n2. YOLO Detection Test...")
try:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    print("   ✓ YOLO model loaded successfully")
    
    # Check if trained model exists
    detect_dir = os.path.join("runs", "detect")
    if os.path.exists(detect_dir):
        runs = [d for d in os.listdir(detect_dir) if d.startswith("mask_finetune")]
        if runs:
            print(f"   ✓ Found {len(runs)} training run(s)")
        else:
            print("   ⚠ No trained models found (default YOLO only)")
    else:
        print("   ⚠ No training runs directory")
except Exception as e:
    print(f"   ✗ YOLO failed: {e}")

# 3. ReID Feature Extraction Test
print("\n3. ReID Feature Extraction Test...")
try:
    reid = ReIDExtractor()
    print(f"   ✓ ReID model loaded")
    print(f"   Device: {reid.device}")
    
    # Test feature extraction with dummy data
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 300, 400]
    features = reid.extract_features(dummy_frame, dummy_bbox)
    
    if features is not None:
        print(f"   ✓ Feature extraction working (shape: {features.shape})")
        print(f"   Feature norm: {np.linalg.norm(features):.3f} (should be ~1.0)")
    else:
        print("   ⚠ Feature extraction returned None")
        
except Exception as e:
    print(f"   ✗ ReID failed: {e}")

# 4. Database Test
print("\n4. Database Test...")
try:
    db = ReIDDatabase()
    
    # Count persons
    all_features = db.get_all_features()
    persons = {}
    for f in all_features:
        persons.setdefault(f['person_id'], []).append(f)
    
    print(f"   ✓ Database connected")
    print(f"   Total persons: {len(persons)}")
    print(f"   Total feature vectors: {len(all_features)}")
    
    # Check for unnamed persons
    unnamed_count = 0
    for pid in persons.keys():
        stats = db.get_person_stats(pid)
        if stats and not stats[3]:  # stats[3] is name
            unnamed_count += 1
    
    if unnamed_count > 0:
        print(f"   ⚠ {unnamed_count} person(s) without names")
    
    db.close()
except Exception as e:
    print(f"   ✗ Database failed: {e}")

# 5. GPU Test
print("\n5. GPU Test...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("   ⚠ GPU not available (using CPU)")
    print("   Note: CPU mode will be slower")

# 6. Telegram Configuration Test
print("\n6. Telegram Configuration Test...")
try:
    # Check for .env file
    if os.path.exists(".env"):
        print("   ✓ .env file found")
    else:
        print("   ⚠ .env file not found (copy from .env.example)")
    
    from config.telegram_config import BOT_TOKEN, CHAT_ID
    if BOT_TOKEN and CHAT_ID and BOT_TOKEN != "your_bot_token_here":
        print(f"   ✓ Telegram configured")
        print(f"   Bot token: {BOT_TOKEN[:20]}...")
        print(f"   Chat ID: {CHAT_ID}")
        
        # Optional: test send (commented out by default)
        # import requests
        # r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe")
        # if r.status_code == 200:
        #     print(f"   ✓ Bot token valid: {r.json()['result']['username']}")
        # else:
        #     print(f"   ✗ Bot token invalid or network issue")
    else:
        print("   ⚠ Telegram not configured (check .env file)")
except Exception as e:
    print(f"   ⚠ Telegram config issue: {e}")

# 7. Model Files Check
print("\n7. Model Files Check...")
required_files = ["best.pt"]
for file in required_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   ✓ {file} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {file} missing")

# 8. Directory Structure Check
print("\n8. Directory Structure Check...")
required_dirs = ["config", "database", "detector", "tracker", "tools", "test"]
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/")
    else:
        print(f"   ✗ {dir_name}/ missing")

# 9. Dependencies Check
print("\n9. Key Dependencies Check...")
dependencies = [
    ("cv2", "opencv-python"),
    ("torch", "torch"),
    ("torchreid", "torchreid"),
    ("ultralytics", "ultralytics"),
    ("numpy", "numpy"),
    ("requests", "requests"),
]

for module_name, package_name in dependencies:
    try:
        __import__(module_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name} not installed (pip install {package_name})")

print("\n" + "="*50)
print("Health Check Complete!")
print("="*50)
print("\nNext steps:")
print("1. Fix any ✗ errors above")
print("2. Address ⚠ warnings if needed")
print("3. Enroll persons: python -m tools.enroll_person")
print("4. Run main system: python main.py")
print("5. Check detailed troubleshooting: TROUBLESHOOTING.md")
