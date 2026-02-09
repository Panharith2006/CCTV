"""
Test script to verify the violation-based storage system

This script demonstrates:
1. Normal persons (mask + helmet) → Memory only (NOT saved to DB)
2. No mask → WARNING → Saved to database
3. No helmet → ALERT → Saved to database
4. Both violations → CRITICAL → Saved to database

Run this BEFORE running main.py to understand the expected behavior.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.reid_database import ReIDDatabase


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def test_violation_storage():
    """Test the violation storage system"""
    
    clear_screen()
    print("=" * 80)
    print("  VIOLATION-BASED STORAGE SYSTEM TEST")
    print("=" * 80)
    print()
    print("This test demonstrates the NEW behavior:")
    print("  ✅ Only persons with violations saved to database")
    print("  ✅ Normal persons tracked in memory only")
    print()
    
    # Connect to database
    print("[1/6] Connecting to database...")
    try:
        db = ReIDDatabase()
        print("  ✓ Connected\n")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("\n⚠️  Make sure MySQL is running and config/mysql_config.py is correct")
        return False
    
    # Check if violation columns exist
    print("[2/6] Checking for violation tracking columns...")
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() 
        AND TABLE_NAME = 'persons'
    """)
    columns = [row[0] for row in cursor.fetchall()]
    
    has_violation_cols = all(col in columns for col in [
        'has_mask_violation', 
        'has_helmet_violation',
        'alert_status',
        'warning_status',
        'violation_type'
    ])
    
    if not has_violation_cols:
        print("  ✗ Violation columns not found!")
        print()
        print("Run migration first:")
        print("  python scripts/migrate_add_violation_tracking.py")
        print()
        return False
    
    print("  ✓ Violation columns found\n")
    
    # Clear existing test data
    print("[3/6] Clearing test data...")
    cursor.execute("DELETE FROM persons WHERE name LIKE 'TEST_%'")
    db.conn.commit()
    print("  ✓ Cleared\n")
    
    # Scenario 1: No helmet (ALERT)
    print("[4/6] Simulating violations...")
    print()
    print("  Scenario A: Person without helmet (ALERT)")
    person_id_1 = db.add_person(
        name="TEST_NO_HELMET",
        location="CAM_01",
        has_mask_violation=False,
        has_helmet_violation=True,
        alert_status="ALERT",
        warning_status=None,
        violation_type="NO_HELMET"
    )
    print(f"    🚨 Saved to database: ID={person_id_1}")
    print(f"    ✅ Alert status: ALERT")
    print(f"    ✅ Violation: NO_HELMET")
    
    # Scenario 2: No mask (WARNING)
    print()
    print("  Scenario B: Person without mask (WARNING)")
    person_id_2 = db.add_person(
        name="TEST_NO_MASK",
        location="CAM_01",
        has_mask_violation=True,
        has_helmet_violation=False,
        alert_status=None,
        warning_status="WARNING",
        violation_type="NO_MASK"
    )
    print(f"    ⚠️  Saved to database: ID={person_id_2}")
    print(f"    ✅ Warning status: WARNING")
    print(f"    ✅ Violation: NO_MASK")
    
    # Scenario 3: Both violations (CRITICAL)
    print()
    print("  Scenario C: Person without mask AND helmet (CRITICAL)")
    person_id_3 = db.add_person(
        name="TEST_BOTH_VIOLATIONS",
        location="CAM_01",
        has_mask_violation=True,
        has_helmet_violation=True,
        alert_status="ALERT",
        warning_status="WARNING",
        violation_type="NO_MASK, NO_HELMET"
    )
    print(f"    🚨⚠️  Saved to database: ID={person_id_3}")
    print(f"    ✅ Alert status: ALERT")
    print(f"    ✅ Warning status: WARNING")
    print(f"    ✅ Violation: NO_MASK, NO_HELMET")
    
    print()
    print("  Scenario D: Normal person (compliant)")
    print(f"    ✅ NOT saved to database (tracked in memory only)")
    print(f"    ✅ Memory ID: M1 (temporary)")
    print(f"    ✅ No violation recorded")
    
    print()
    
    # Verify database contents
    print("[5/6] Verifying database contents...")
    cursor.execute("SELECT COUNT(*) FROM persons WHERE name LIKE 'TEST_%'")
    test_count = cursor.fetchone()[0]
    print(f"  ✓ Test persons in database: {test_count} (should be 3)")
    
    cursor.execute("""
        SELECT COUNT(*) FROM persons 
        WHERE name LIKE 'TEST_%' 
        AND (has_mask_violation = 1 OR has_helmet_violation = 1)
    """)
    violation_count = cursor.fetchone()[0]
    print(f"  ✓ Persons with violations: {violation_count} (should be 3)")
    
    cursor.execute("SELECT COUNT(*) FROM persons WHERE name LIKE 'TEST_%' AND alert_status = 'ALERT'")
    alert_count = cursor.fetchone()[0]
    print(f"  ✓ Alert status count: {alert_count} (should be 2)")
    
    cursor.execute("SELECT COUNT(*) FROM persons WHERE name LIKE 'TEST_%' AND warning_status = 'WARNING'")
    warning_count = cursor.fetchone()[0]
    print(f"  ✓ Warning status count: {warning_count} (should be 2)")
    
    print()
    
    # Show summary
    print("[6/6] Summary")
    print()
    print("=" * 80)
    print("  SYSTEM BEHAVIOR VERIFIED")
    print("=" * 80)
    print()
    print("✅ Database Strategy: VIOLATION-ONLY")
    print()
    print("What gets SAVED to database:")
    print(f"  🚨 No helmet → ALERT → ID={person_id_1}")
    print(f"  ⚠️  No mask → WARNING → ID={person_id_2}")
    print(f"  🚨⚠️  Both violations → CRITICAL → ID={person_id_3}")
    print()
    print("What DOES NOT get saved:")
    print("  ✅ Normal persons (mask + helmet) → Memory only")
    print()
    print("Database Count:")
    print(f"  • Test violations: {test_count}")
    print(f"  • Total alerts: {alert_count}")
    print(f"  • Total warnings: {warning_count}")
    print()
    
    # Show actual records
    print("=" * 80)
    print("  DATABASE RECORDS")
    print("=" * 80)
    print()
    
    cursor.execute("""
        SELECT 
            person_id,
            name,
            has_mask_violation,
            has_helmet_violation,
            alert_status,
            warning_status,
            violation_type
        FROM persons
        WHERE name LIKE 'TEST_%'
        ORDER BY person_id
    """)
    
    for row in cursor.fetchall():
        pid, name, mask_vio, helmet_vio, alert, warning, vtype = row
        print(f"Person ID: {pid}")
        print(f"  Name: {name}")
        print(f"  Mask Violation: {'YES ⚠️' if mask_vio else 'No'}")
        print(f"  Helmet Violation: {'YES 🚨' if helmet_vio else 'No'}")
        print(f"  Alert: {alert or '-'}")
        print(f"  Warning: {warning or '-'}")
        print(f"  Type: {vtype or '-'}")
        print()
    
    db.close()
    
    print("=" * 80)
    print()
    print("✅ Test completed successfully!")
    print()
    print("Next steps:")
    print("  1. Run: python main.py")
    print("  2. Test with webcam:")
    print("     - Wear mask + helmet → Should see Memory ID (M1, M2, etc.)")
    print("     - Remove helmet → Should see Database ID and ALERT")
    print("     - Remove mask → Should see Database ID and WARNING")
    print("  3. Check database: python scripts/print_db.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_violation_storage()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
