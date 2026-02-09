"""
Migration Script: Add Violation Tracking & Alert/Warning Status

This migration adds:
1. Violation tracking columns: has_mask_violation, has_helmet_violation
2. Alert/Warning status columns: alert_status, warning_status, violation_type
3. Changes database strategy to ONLY save persons with violations

Critical Change:
- OLD: Save all detected persons to database
- NEW: Save ONLY persons with mask/helmet violations
- Normal persons tracked in memory only
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mysql_config import MYSQL_CONFIG
import mysql.connector


def migrate():
    """Add violation tracking and alert/warning status to database"""
    print("\n" + "="*80)
    print("  Migration: Add Violation Tracking & Alert/Warning Status")
    print("="*80 + "\n")
    
    print("⚠️  CRITICAL SYSTEM CHANGE:")
    print("   OLD: All detected persons saved to database")
    print("   NEW: ONLY persons with violations saved to database")
    print()
    print("   • No mask = WARNING (saved)")
    print("   • No helmet = ALERT (saved)")
    print("   • Normal persons = Memory only (NOT saved)")
    print()
    
    try:
        # Connect to MySQL
        print(f"[1/5] Connecting to MySQL database: {MYSQL_CONFIG['database']}")
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
        
        # Step 1: Check existing columns
        print("[2/5] Checking existing schema...")
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'persons'
        """, (MYSQL_CONFIG['database'],))
        existing_cols = [row[0] for row in cursor.fetchall()]
        
        # Step 2: Add new violation tracking columns
        print("[3/5] Adding violation tracking columns to persons table...")
        
        new_columns = {
            'has_mask_violation': 'TINYINT(1) DEFAULT 0 COMMENT "1 if person detected without mask"',
            'has_helmet_violation': 'TINYINT(1) DEFAULT 0 COMMENT "1 if person detected without helmet"',
            'alert_status': 'VARCHAR(50) DEFAULT NULL COMMENT "ALERT if no helmet"',
            'warning_status': 'VARCHAR(50) DEFAULT NULL COMMENT "WARNING if no mask"',
            'violation_type': 'VARCHAR(100) DEFAULT NULL COMMENT "NO_MASK, NO_HELMET, or both"'
        }
        
        for col_name, col_def in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE persons ADD COLUMN {col_name} {col_def}")
                print(f"  ✓ Added column: {col_name}")
            else:
                print(f"  • Column already exists: {col_name}")
        
        conn.commit()
        print()
        
        # Step 3: Add indexes for violation queries
        print("[4/5] Adding indexes for violation queries...")
        try:
            cursor.execute("""
                CREATE INDEX idx_violations 
                ON persons(has_mask_violation, has_helmet_violation)
            """)
            print("  ✓ Created idx_violations index")
        except Exception as e:
            if "Duplicate" in str(e):
                print("  • Index already exists: idx_violations")
            else:
                print(f"  ⚠️  Could not create index: {e}")
        
        conn.commit()
        print()
        
        # Step 4: Update existing persons (mark as normal if no violation data)
        print("[5/5] Updating existing persons...")
        cursor.execute("""
            UPDATE persons 
            SET has_mask_violation = 0, 
                has_helmet_violation = 0,
                violation_type = 'LEGACY_DATA'
            WHERE has_mask_violation IS NULL 
            OR has_helmet_violation IS NULL
        """)
        
        updated = cursor.rowcount
        if updated > 0:
            print(f"  ✓ Updated {updated} legacy person records")
        else:
            print("  • No legacy data to update")
        
        conn.commit()
        print()
        
        # Show summary
        print("="*80)
        print("Migration Summary:")
        print("-" * 80)
        
        cursor.execute("SELECT COUNT(*) FROM persons")
        person_count = cursor.fetchone()[0]
        print(f"  • Total persons in database: {person_count}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM persons 
            WHERE has_mask_violation = 1 OR has_helmet_violation = 1
        """)
        violation_count = cursor.fetchone()[0]
        print(f"  • Persons with violations: {violation_count}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM persons 
            WHERE alert_status = 'ALERT'
        """)
        alert_count = cursor.fetchone()[0]
        print(f"  • Alert status (no helmet): {alert_count}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM persons 
            WHERE warning_status = 'WARNING'
        """)
        warning_count = cursor.fetchone()[0]
        print(f"  • Warning status (no mask): {warning_count}")
        
        print()
        print("✓ Migration completed successfully!")
        print()
        print("New System Behavior:")
        print("  1. ✓ Only persons with violations saved to database")
        print("  2. ✓ Alert status: No helmet detected (CRITICAL)")
        print("  3. ✓ Warning status: No mask detected (CAUTION)")
        print("  4. ✓ Normal persons tracked in memory only")
        print("  5. ✓ Database focused on violations only")
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
    print("This migration will change the database strategy:")
    print("  • FROM: Save all detected persons")
    print("  • TO: Save ONLY persons with violations (mask/helmet)")
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
        print("Your database is now configured for violation-only tracking!")
        print("Run: python main.py")
        print()
    
    sys.exit(0 if success else 1)
