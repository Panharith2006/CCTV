import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mysql_config import MYSQL_CONFIG
import mysql.connector


def migrate():
    """Add is_masked and is_helmeted columns to existing features table"""
    print("\n" + "="*60)
    print("  Migration: Add is_masked & is_helmeted columns")
    print("="*60 + "\n")
    
    try:
        # Connect to MySQL
        print(f"[1/3] Connecting to MySQL database: {MYSQL_CONFIG['database']}")
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
        
        # Check if columns already exist
        print("[2/3] Checking existing schema...")
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'features'
            AND COLUMN_NAME IN ('is_masked', 'is_helmeted')
        """, (MYSQL_CONFIG['database'],))
        existing_cols = [row[0] for row in cursor.fetchall()]
        
        if 'is_masked' in existing_cols and 'is_helmeted' in existing_cols:
            print("✓ Columns already exist - no migration needed\n")
            cursor.close()
            conn.close()
            return True
        
        # Add missing columns
        print("[3/3] Adding columns to features table...")
        
        if 'is_masked' not in existing_cols:
            cursor.execute("""
                ALTER TABLE features 
                ADD COLUMN is_masked TINYINT(1) DEFAULT 0
            """)
            print("  ✓ Added is_masked column")
        
        if 'is_helmeted' not in existing_cols:
            cursor.execute("""
                ALTER TABLE features 
                ADD COLUMN is_helmeted TINYINT(1) DEFAULT 0
            """)
            print("  ✓ Added is_helmeted column")
        
        conn.commit()
        print("\n✓ Migration completed successfully!\n")
        
        # Show summary
        cursor.execute("SELECT COUNT(*) FROM features")
        feature_count = cursor.fetchone()[0]
        print(f"  Total features in database: {feature_count}")
        print(f"  All existing features marked as: masked=False, helmeted=False")
        print(f"  New captures can now specify masked/helmeted variants\n")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}\n")
        return False


if __name__ == "__main__":
    success = migrate()
    if success:
        print("You can now use the updated enrollment tool to capture")
        print("masked and helmeted variants for each person.\n")
        print("Example usage:")
        print("  python -m tools.enroll_person")
        print("  Then select option 2 (masked), 3 (helmeted), or 6 (add to existing)\n")
    sys.exit(0 if success else 1)
