"""
Project Verification Script
Checks if all ReID components are properly set up
"""

import sys
import os

def check_imports():
    """Check if all required modules can be imported"""
    print("="*60)
    print("CHECKING IMPORTS")
    print("="*60)
    
    modules = [
        ("torch", "PyTorch"),
        ("torchreid", "TorchReID"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("sqlite3", "SQLite3"),
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - MISSING: {e}")
            all_ok = False
    
    return all_ok


def check_files():
    """Check if all required files exist"""
    print("\n" + "="*60)
    print("CHECKING PROJECT FILES")
    print("="*60)
    
    required_files = [
        "detector/layer2_reid_extractor.py",
        "database/reid_database.py",
        "tracker/layer3_reid_manager.py",
        "test/test_reid.py",
        "test/test_reid_full.py",
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_ok = False
    
    return all_ok


def check_reid_system():
    """Test if ReID system can be initialized"""
    print("\n" + "="*60)
    print("CHECKING ReID SYSTEM")
    print("="*60)
    
    try:
        from detector.layer2_reid_extractor import ReIDExtractor
        print("✓ ReIDExtractor can be imported")
        
        extractor = ReIDExtractor()
        print(f"✓ ReIDExtractor initialized successfully")
        print(f"  Device: {extractor.device}")
        print(f"  Model: {extractor.model.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"✗ ReID system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_database():
    """Test if database can be created"""
    print("\n" + "="*60)
    print("CHECKING DATABASE")
    print("="*60)
    
    try:
        from database.reid_database import ReIDDatabase
        print("✓ ReIDDatabase can be imported")
        
        db = ReIDDatabase("test_verification.db")
        print("✓ Database created successfully")
        
        # Clean up test database
        db.close()
        if os.path.exists("test_verification.db"):
            os.remove("test_verification.db")
            print("✓ Database cleanup complete")
        
        return True
    except Exception as e:
        print(f"✗ Database system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_reid_manager():
    """Test if ReID manager can be initialized"""
    print("\n" + "="*60)
    print("CHECKING ReID MANAGER")
    print("="*60)
    
    try:
        from tracker.layer3_reid_manager import ReIDManager
        print("✓ ReIDManager can be imported")
        
        manager = ReIDManager("test_verification.db", similarity_threshold=0.7)
        print("✓ ReIDManager initialized successfully")
        print(f"  Threshold: {manager.similarity_threshold}")
        
        # Clean up
        manager.close()
        if os.path.exists("test_verification.db"):
            os.remove("test_verification.db")
        
        return True
    except Exception as e:
        print(f"✗ ReID manager failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "PROJECT VERIFICATION" + " "*23 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    results = []
    
    # Run all checks
    results.append(("Imports", check_imports()))
    results.append(("Files", check_files()))
    results.append(("ReID System", check_reid_system()))
    results.append(("Database", check_database()))
    results.append(("ReID Manager", check_reid_manager()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\nYour project is ready! Next steps:")
        print("  1. Run: python -m test.test_reid")
        print("  2. Run: python -m test.test_reid_full")
        print("  3. Integrate ReID into main.py")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
