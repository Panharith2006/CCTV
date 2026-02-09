"""
Migration Script: Add Location Tracking & Suspect Management to ReID Database

This migration adds:
1. New columns to persons table: last_camera_id, last_location, suspect_reason, number_of_sequences_saved
2. New table: location_history (track movement across cameras)
3. New table: suspect_images (store images for suspects only)

Run this script to update your existing database to support the full ReID system.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mysql_config import MYSQL_CONFIG
import mysql.connector


def migrate():
    """Add location tracking and suspect management to ReID database"""
    print("\n" + "="*80)
    print("  Migration: Add Location Tracking & Suspect Management")
    print("="*80 + "\n")
    
    try:
        # Connect to MySQL
        print(f"[1/6] Connecting to MySQL database: {MYSQL_CONFIG['database']}")
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG.get('port', 3306),
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
            auth_plugin=MYSQL_CONFIG.get('auth_plugin', 'mysql_native_password')
        )
        cursor = conn.cursor()
        print("✓ Connected\n")
        
        # Step 1: Check and add columns to persons table
        print("[2/6] Updating persons table...")
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'persons'
        """, (MYSQL_CONFIG['database'],))
        existing_cols = [row[0] for row in cursor.fetchall()]
        
        new_columns = {
            'last_camera_id': 'VARCHAR(100) DEFAULT NULL',
            'last_location': 'VARCHAR(255) DEFAULT NULL',
            'suspect_reason': 'VARCHAR(512) DEFAULT NULL',
            'number_of_sequences_saved': 'INT DEFAULT 1'
        }
        
        for col_name, col_def in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE persons ADD COLUMN {col_name} {col_def}")
                print(f"  ✓ Added column: {col_name}")
            else:
                print(f"  • Column already exists: {col_name}")
        
        conn.commit()
        print()
        
        # Step 2: Create location_history table
        print("[3/6] Creating location_history table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_history (
                history_id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT,
                camera_id VARCHAR(100),
                location VARCHAR(255),
                timestamp DATETIME,
                is_masked TINYINT(1) DEFAULT 0,
                is_helmeted TINYINT(1) DEFAULT 0,
                event_type VARCHAR(50) DEFAULT 'movement',
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_timestamp (person_id, timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("  ✓ location_history table created/verified\n")
        conn.commit()
        
        # Step 3: Create suspect_images table
        print("[4/6] Creating suspect_images table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suspect_images (
                image_id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT,
                image_path VARCHAR(1024),
                timestamp DATETIME,
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_id (person_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("  ✓ suspect_images table created/verified\n")
        conn.commit()
        
        # Step 4: Populate location_history from existing data
        print("[5/6] Populating location_history from existing features...")
        cursor.execute("""
            SELECT person_id, camera_id, timestamp, is_masked, is_helmeted
            FROM features
            ORDER BY person_id, timestamp
        """)
        
        features = cursor.fetchall()
        if features:
            # Group by person and add first detection
            from collections import defaultdict
            person_features = defaultdict(list)
            for pid, cam, ts, masked, helmeted in features:
                person_features[pid].append((cam, ts, masked, helmeted))
            
            for pid, feat_list in person_features.items():
                # Check if location history already exists
                cursor.execute("SELECT COUNT(*) FROM location_history WHERE person_id = %s", (pid,))
                if cursor.fetchone()[0] == 0:
                    # Add first detection
                    cam, ts, masked, helmeted = feat_list[0]
                    cursor.execute("""
                        INSERT INTO location_history (person_id, camera_id, location, timestamp, is_masked, is_helmeted, event_type)
                        VALUES (%s, %s, %s, %s, %s, %s, 'first_detection')
                    """, (pid, cam, 'Unknown Location', ts, masked, helmeted))
            
            conn.commit()
            print(f"  ✓ Added location history for {len(person_features)} persons\n")
        else:
            print("  • No existing data to migrate\n")
        
        # Step 5: Show summary
        print("[6/6] Migration Summary:")
        print("-" * 80)
        
        cursor.execute("SELECT COUNT(*) FROM persons")
        person_count = cursor.fetchone()[0]
        print(f"  • Total persons: {person_count}")
        
        cursor.execute("SELECT COUNT(*) FROM features")
        feature_count = cursor.fetchone()[0]
        print(f"  • Total features: {feature_count}")
        
        cursor.execute("SELECT COUNT(*) FROM location_history")
        location_count = cursor.fetchone()[0]
        print(f"  • Location history records: {location_count}")
        
        cursor.execute("SELECT COUNT(*) FROM suspect_images")
        suspect_img_count = cursor.fetchone()[0]
        print(f"  • Suspect images: {suspect_img_count}")
        
        print()
        print("✓ Migration completed successfully!")
        print()
        print("New Features Available:")
        print("  1. ✓ Cross-camera tracking (location history)")
        print("  2. ✓ Suspect person flagging")
        print("  3. ✓ Suspect image storage")
        print("  4. ✓ Scene exit detection")
        print("  5. ✓ Controlled saving strategy (first, suspect, exit only)")
        print()
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("This migration will add location tracking and suspect management features")
    print("to your existing ReID database.")
    print()
    
    try:
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Migration cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nMigration cancelled.")
        sys.exit(0)
    
    success = migrate()
    
    if success:
        print("Your database is now ready for full ReID tracking!")
        print("You can now run: python main.py")
        print()
    
    sys.exit(0 if success else 1)
