"""
Database Migration Script: OLD System → REVISED System

This script migrates the database from the old schema to the new revised schema.

WHAT THIS DOES:
1. Adds new columns: violation_status, violation_reason, detection_date, is_reidentified
2. Removes old columns: has_mask_violation, has_helmet_violation, alert_status, warning_status, violation_type, suspect_reason, number_of_sequences_saved
3. Migrates existing data to new format
4. Updates indexes for performance

⚠️  IMPORTANT: This is a ONE-WAY migration. Backup your database first!

USAGE:
    python scripts/migrate_to_revised_system.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import config
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import mysql.connector
from mysql.connector import Error
from datetime import datetime


def column_exists(cursor, table_name, column_name, database_name):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = %s 
        AND COLUMN_NAME = %s
    """, (database_name, table_name, column_name))
    return cursor.fetchone()[0] > 0


def migrate_database():
    print("=" * 60)
    print("DATABASE MIGRATION: OLD → REVISED SYSTEM")
    print("=" * 60)
    print()
    
    # Import config
    try:
        from config.mysql_config import MYSQL_CONFIG
        config = MYSQL_CONFIG
    except ImportError as e:
        print("❌ ERROR: Could not import MySQL config")
        print(f"   Details: {e}")
        print("   Make sure config/mysql_config.py exists with MYSQL_CONFIG defined")
        return
    
    # Connect to database
    try:
        conn = mysql.connector.connect(
            host=config['host'],
            port=config.get('port', 3306),
            user=config['user'],
            password=config['password'],
            database=config['database'],
            charset=config.get('charset', 'utf8mb4'),
            auth_plugin=config.get('auth_plugin', 'mysql_native_password')
        )
        cursor = conn.cursor()
        print(f"✅ Connected to database: {config['database']}")
    except Error as e:
        print(f"❌ ERROR: Could not connect to database: {e}")
        return
    
    print()
    print("⚠️  WARNING: This will modify your database schema!")
    print("   Make sure you have a backup before proceeding.")
    print()
    response = input("Continue with migration? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Migration cancelled")
        return
    
    print()
    print("Starting migration...")
    print()
    
    # Step 1: Add new columns
    print("Step 1: Adding new columns...")
    
    database_name = config['database']
    
    # Check and add violation_status
    if not column_exists(cursor, 'persons', 'violation_status', database_name):
        try:
            cursor.execute("""
                ALTER TABLE persons
                ADD COLUMN violation_status VARCHAR(20) DEFAULT NULL
            """)
            conn.commit()
            print("  ✅ Added violation_status")
        except Error as e:
            print(f"  ❌ Error adding violation_status: {e}")
            cursor.close()
            conn.close()
            return
    else:
        print("  ℹ️  violation_status already exists")
    
    # Check and add violation_reason
    if not column_exists(cursor, 'persons', 'violation_reason', database_name):
        try:
            cursor.execute("""
                ALTER TABLE persons
                ADD COLUMN violation_reason VARCHAR(255) DEFAULT NULL
            """)
            conn.commit()
            print("  ✅ Added violation_reason")
        except Error as e:
            print(f"  ❌ Error adding violation_reason: {e}")
            cursor.close()
            conn.close()
            return
    else:
        print("  ℹ️  violation_reason already exists")
    
    # Check and add detection_date
    if not column_exists(cursor, 'persons', 'detection_date', database_name):
        try:
            cursor.execute("""
                ALTER TABLE persons
                ADD COLUMN detection_date DATE
            """)
            conn.commit()
            print("  ✅ Added detection_date")
        except Error as e:
            print(f"  ❌ Error adding detection_date: {e}")
            cursor.close()
            conn.close()
            return
    else:
        print("  ℹ️  detection_date already exists")
    
    # Check and add is_reidentified
    if not column_exists(cursor, 'persons', 'is_reidentified', database_name):
        try:
            cursor.execute("""
                ALTER TABLE persons
                ADD COLUMN is_reidentified TINYINT(1) DEFAULT 0
            """)
            conn.commit()
            print("  ✅ Added is_reidentified")
        except Error as e:
            print(f"  ❌ Error adding is_reidentified: {e}")
            cursor.close()
            conn.close()
            return
    else:
        print("  ℹ️  is_reidentified already exists")
    
    print()
    
    # Step 2: Migrate existing data
    print("Step 2: Migrating existing data...")
    
    # Check if old columns exist first
    old_columns_exist = True
    required_old_columns = ['has_mask_violation', 'has_helmet_violation', 'alert_status', 'warning_status', 'violation_type', 'first_seen']
    
    for col in required_old_columns:
        if not column_exists(cursor, 'persons', col, database_name):
            print(f"  ℹ️  Old column '{col}' doesn't exist - this might be a fresh install")
            old_columns_exist = False
            break
    
    if not old_columns_exist:
        print("  ℹ️  Skipping data migration - no old data to migrate")
        print("  ℹ️  This appears to be a fresh database installation")
    else:
        # Get all persons
        cursor.execute("SELECT person_id, has_mask_violation, has_helmet_violation, alert_status, warning_status, violation_type, first_seen FROM persons")
        persons = cursor.fetchall()
        
        migrated_count = 0
        for person in persons:
            person_id, has_mask, has_helmet, alert_status, warning_status, violation_type, first_seen = person
            
            # Determine new violation_status and violation_reason
            violation_status = None
            violation_reason = None
            
            if has_helmet:
                violation_status = "ALERT"
                violation_reason = "HELMET"
            elif has_mask:
                violation_status = "WARNING"
                violation_reason = "MASK"
            elif alert_status:
                violation_status = "ALERT"
                violation_reason = violation_type if violation_type else "UNKNOWN"
            elif warning_status:
                violation_status = "WARNING"
                violation_reason = violation_type if violation_type else "UNKNOWN"
            
            # Both mask and helmet
            if has_mask and has_helmet:
                violation_status = "ALERT"
                violation_reason = "MASK+HELMET"
            
            # Set detection date from first_seen
            detection_date = first_seen.date() if first_seen else datetime.now().date()
            
            # Update record
            cursor.execute("""
                UPDATE persons
                SET violation_status = %s,
                    violation_reason = %s,
                    detection_date = %s
                WHERE person_id = %s
            """, (violation_status, violation_reason, detection_date, person_id))
            
            migrated_count += 1
        
        conn.commit()
        print(f"  ✅ Migrated {migrated_count} person records")
    
    print()
    
    # Step 3: Remove old columns (optional - commented out for safety)
    print("Step 3: Removing old columns...")
    print("  ⚠️  SKIPPED - Remove old columns manually if desired:")
    print("     ALTER TABLE persons DROP COLUMN has_mask_violation;")
    print("     ALTER TABLE persons DROP COLUMN has_helmet_violation;")
    print("     ALTER TABLE persons DROP COLUMN alert_status;")
    print("     ALTER TABLE persons DROP COLUMN warning_status;")
    print("     ALTER TABLE persons DROP COLUMN violation_type;")
    print("     ALTER TABLE persons DROP COLUMN suspect_reason;")
    print("     ALTER TABLE persons DROP COLUMN number_of_sequences_saved;")
    print()
    
    # Uncomment to actually remove old columns (DANGEROUS!)
    # try:
    #     cursor.execute("ALTER TABLE persons DROP COLUMN has_mask_violation")
    #     cursor.execute("ALTER TABLE persons DROP COLUMN has_helmet_violation")
    #     cursor.execute("ALTER TABLE persons DROP COLUMN alert_status")
    #     cursor.execute("ALTER TABLE persons DROP COLUMN warning_status")
    #     cursor.execute("ALTER TABLE persons DROP COLUMN violation_type")
    #     cursor.execute("ALTER TABLE persons DROP COLUMN suspect_reason")
    #     cursor.execute("ALTER TABLE persons DROP COLUMN number_of_sequences_saved")
    #     conn.commit()
    #     print("  ✅ Removed old columns")
    # except Error as e:
    #     print(f"  ❌ Error removing columns: {e}")
    
    # Step 4: Update indexes
    print("Step 4: Updating indexes...")
    
    # Check if old index exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.STATISTICS 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = 'persons' 
        AND INDEX_NAME = 'idx_violations'
    """, (database_name,))
    
    if cursor.fetchone()[0] > 0:
        try:
            cursor.execute("DROP INDEX idx_violations ON persons")
            conn.commit()
            print("  ✅ Removed old idx_violations")
        except Error as e:
            print(f"  ⚠️  Error removing idx_violations: {e}")
    else:
        print("  ℹ️  idx_violations doesn't exist")
    
    # Check if new violation index exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.STATISTICS 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = 'persons' 
        AND INDEX_NAME = 'idx_violation'
    """, (database_name,))
    
    if cursor.fetchone()[0] == 0:
        try:
            cursor.execute("CREATE INDEX idx_violation ON persons(violation_status)")
            conn.commit()
            print("  ✅ Created idx_violation")
        except Error as e:
            print(f"  ❌ Error creating idx_violation: {e}")
    else:
        print("  ℹ️  idx_violation already exists")
    
    # Check if detection_date index exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.STATISTICS 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = 'persons' 
        AND INDEX_NAME = 'idx_detection_date'
    """, (database_name,))
    
    if cursor.fetchone()[0] == 0:
        try:
            cursor.execute("CREATE INDEX idx_detection_date ON persons(detection_date)")
            conn.commit()
            print("  ✅ Created idx_detection_date")
        except Error as e:
            print(f"  ❌ Error creating idx_detection_date: {e}")
    else:
        print("  ℹ️  idx_detection_date already exists")
    
    print()
    
    # Step 5: Verify migration
    print("Step 5: Verifying migration...")
    
    cursor.execute("SELECT COUNT(*) FROM persons WHERE violation_status IS NOT NULL")
    violations_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM persons WHERE detection_date IS NOT NULL")
    dates_count = cursor.fetchone()[0]
    
    print(f"  ✅ Persons with violation_status: {violations_count}")
    print(f"  ✅ Persons with detection_date: {dates_count}")
    print()
    
    # Close connection
    cursor.close()
    conn.close()
    
    print("=" * 60)
    print("✅ MIGRATION COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Verify data integrity in database")
    print("2. Test system with new schema")
    print("3. If everything works, manually remove old columns (see Step 3)")
    print()


if __name__ == "__main__":
    migrate_database()
