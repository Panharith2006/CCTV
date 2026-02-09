"""
Test MySQL Connection and ReID Database Setup

This script:
1. Tests MySQL connection
2. Verifies all required tables exist
3. Shows database statistics
4. Tests basic database operations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mysql_connection():
    """Test MySQL connection and database setup"""
    print("\n" + "="*80)
    print("  ReID Database Connection Test")
    print("="*80 + "\n")
    
    try:
        # Test 1: Import MySQL config
        print("[1/5] Loading MySQL configuration...")
        try:
            from config.mysql_config import MYSQL_CONFIG
            print(f"  ✓ Config loaded: {MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}")
            print(f"  • Database: {MYSQL_CONFIG['database']}")
            
            if not MYSQL_CONFIG['password']:
                print(f"  ⚠️  WARNING: Password is empty - make sure this is intentional")
        except ImportError as e:
            print(f"  ✗ Failed to import config: {e}")
            print("\n  Create config/mysql_config.py with your MySQL credentials:")
            print("""
    MYSQL_CONFIG = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'root1234',
        'database': 'cctv_ai',
        'charset': 'utf8mb4',
        'autocommit': False
    }
            """)
            return False
        
        print()
        
        # Test 2: Test connection
        print("[2/5] Testing MySQL connection...")
        try:
            from database.reid_database import ReIDDatabase
            db = ReIDDatabase(MYSQL_CONFIG)
            print("  ✓ Connected successfully!")
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            print("\n  Troubleshooting:")
            print("  1. Make sure MySQL server is running")
            print("  2. Verify username and password in config/mysql_config.py")
            print("  3. Check if user has permission to create databases")
            print("  4. Try: mysql -u root -p")
            return False
        
        print()
        
        # Test 3: Verify tables
        print("[3/5] Verifying database tables...")
        import mysql.connector
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG.get('port', 3306),
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database']
        )
        cursor = conn.cursor()
        
        required_tables = ['persons', 'features', 'detections', 'location_history', 'suspect_images']
        cursor.execute("SHOW TABLES")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        all_exist = True
        for table in required_tables:
            if table in existing_tables:
                print(f"  ✓ Table exists: {table}")
            else:
                print(f"  ✗ Table missing: {table}")
                all_exist = False
        
        if not all_exist:
            print("\n  ⚠️  Some tables are missing!")
            print("  Run migration script: python scripts/migrate_add_location_tracking.py")
            return False
        
        print()
        
        # Test 4: Check table structure
        print("[4/5] Verifying table columns...")
        
        # Check persons table
        cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'persons'
        """, (MYSQL_CONFIG['database'],))
        person_cols = [row[0] for row in cursor.fetchall()]
        
        required_person_cols = ['person_id', 'first_seen', 'last_seen', 'status', 
                               'last_camera_id', 'last_location', 'suspect_reason', 
                               'number_of_sequences_saved']
        
        missing_cols = [col for col in required_person_cols if col not in person_cols]
        if missing_cols:
            print(f"  ✗ persons table missing columns: {missing_cols}")
            print("  Run migration script: python scripts/migrate_add_location_tracking.py")
            return False
        else:
            print(f"  ✓ persons table has all required columns")
        
        # Check features table
        cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'features'
        """, (MYSQL_CONFIG['database'],))
        feature_cols = [row[0] for row in cursor.fetchall()]
        
        if 'is_masked' in feature_cols and 'is_helmeted' in feature_cols:
            print(f"  ✓ features table has mask/helmet columns")
        else:
            print(f"  ✗ features table missing mask/helmet columns")
            print("  Run migration script: python scripts/migrate_add_mask_helmet_columns.py")
            return False
        
        print()
        
        # Test 5: Show statistics
        print("[5/5] Database Statistics:")
        print("-" * 80)
        
        summary = db.get_summary()
        print(f"  • Total persons: {summary['total_persons']}")
        print(f"  • Active persons: {summary['active_persons']}")
        print(f"  • Suspects: {summary['suspects']}")
        print(f"  • Exited: {summary['exited']}")
        print(f"  • Suspect images: {summary['total_suspect_images']}")
        
        cursor.execute("SELECT COUNT(*) FROM features")
        feature_count = cursor.fetchone()[0]
        print(f"  • Feature vectors: {feature_count}")
        
        cursor.execute("SELECT COUNT(*) FROM location_history")
        location_count = cursor.fetchone()[0]
        print(f"  • Location records: {location_count}")
        
        print()
        print("="*80)
        print("✓ All tests passed! Your ReID database is ready.")
        print("="*80)
        print()
        
        cursor.close()
        conn.close()
        db.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_operations():
    """Test basic database operations"""
    print("\n" + "="*80)
    print("  Testing Basic Database Operations")
    print("="*80 + "\n")
    
    try:
        from database.reid_database import ReIDDatabase
        from config.mysql_config import MYSQL_CONFIG
        import numpy as np
        
        db = ReIDDatabase(MYSQL_CONFIG)
        
        # Test: Add person
        print("[1/3] Testing add_person()...")
        test_vector = np.random.rand(512)
        person_id = db.add_person(
            feature_vector=test_vector,
            camera_id="TEST_CAM",
            location="Test Location",
            is_masked=False,
            is_helmeted=False
        )
        print(f"  ✓ Added test person: ID={person_id}")
        
        # Test: Update location
        print("\n[2/3] Testing update_person_location()...")
        db.update_person_location(
            person_id=person_id,
            camera_id="TEST_CAM_2",
            location="Test Location 2",
            is_masked=True,
            is_helmeted=False
        )
        print(f"  ✓ Updated location for person {person_id}")
        
        # Test: Mark as suspect
        print("\n[3/3] Testing mark_person_as_suspect()...")
        db.mark_person_as_suspect(person_id, reason="Test suspect")
        print(f"  ✓ Marked person {person_id} as suspect")
        
        # Get location history
        history = db.get_person_location_history(person_id)
        print(f"\n  Location history for person {person_id}:")
        for record in history:
            print(f"    • {record['location']} ({record['camera_id']}) - {record['event_type']}")
        
        # Cleanup test data
        print(f"\n  Cleaning up test person {person_id}...")
        db.delete_person(person_id)
        
        db.close()
        
        print()
        print("="*80)
        print("✓ All operations working correctly!")
        print("="*80)
        print()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("This script will test your MySQL connection and ReID database setup.")
    print()
    
    # Test connection
    if not test_mysql_connection():
        print("\n⚠️  Connection test failed!")
        print("\nNext steps:")
        print("  1. Make sure MySQL server is running")
        print("  2. Configure config/mysql_config.py with your credentials")
        print("  3. Run migration: python scripts/migrate_add_location_tracking.py")
        print()
        sys.exit(1)
    
    # Test operations
    print("\nRunning operation tests...")
    if not test_basic_operations():
        print("\n⚠️  Operation test failed!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("  🎉 SUCCESS! Your ReID system is fully configured!")
    print("="*80)
    print()
    print("You can now run:")
    print("  • python main.py                    - Start CCTV system")
    print("  • python -m tools.enroll_person     - Enroll known persons")
    print("  • python -m tools.review_persons    - Review detected persons")
    print()
    
    sys.exit(0)
