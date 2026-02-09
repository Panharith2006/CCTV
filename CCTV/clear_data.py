"""
Clear all database records and thumbnails for testing
WARNING: This will permanently delete all person data, detections, and thumbnail images!
"""

import os
import mysql.connector
from mysql.connector import Error
from config.mysql_config import MYSQL_CONFIG


def clear_database():
    """Clear all records from database tables"""
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG.get('port', 3306),
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
            autocommit=False,
            auth_plugin=MYSQL_CONFIG.get('auth_plugin', 'mysql_native_password')
        )
        
        cursor = conn.cursor()
        
        # Disable foreign key checks temporarily
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        # List of tables to clear (in order due to foreign key constraints)
        tables = [
            'detections',
            'features',
            'location_history',
            'suspect_images',
            'persons'
        ]
        
        print("[Clear] Clearing database tables...")
        for table in tables:
            try:
                cursor.execute(f"TRUNCATE TABLE {table}")
                print(f"[Clear] ✓ Cleared table: {table}")
            except Error as e:
                print(f"[Clear] ⚠ Warning clearing {table}: {e}")
        
        # Re-enable foreign key checks
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("[Clear] ✓ Database cleared successfully!")
        return True
        
    except Error as e:
        print(f"[Clear] ✗ Error clearing database: {e}")
        return False


def clear_thumbnails():
    """Delete all thumbnail images"""
    thumbnails_dir = "thumbnails"
    
    if not os.path.exists(thumbnails_dir):
        print(f"[Clear] Thumbnails directory not found: {thumbnails_dir}")
        return True
    
    print("[Clear] Clearing thumbnails...")
    deleted_count = 0
    error_count = 0
    
    for filename in os.listdir(thumbnails_dir):
        file_path = os.path.join(thumbnails_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"[Clear] ⚠ Error deleting {filename}: {e}")
                error_count += 1
    
    print(f"[Clear] ✓ Deleted {deleted_count} thumbnail(s)")
    if error_count > 0:
        print(f"[Clear] ⚠ Failed to delete {error_count} file(s)")
    
    return error_count == 0


def main():
    print("=" * 60)
    print("CLEAR DATABASE AND THUMBNAILS FOR TESTING")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This will permanently delete:")
    print("   - All person records")
    print("   - All feature vectors")
    print("   - All detections")
    print("   - All location history")
    print("   - All suspect images records")
    print("   - All thumbnail images")
    print()
    
    # Confirm action
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("[Clear] Operation cancelled.")
        return
    
    print()
    
    # Clear database
    db_success = clear_database()
    print()
    
    # Clear thumbnails
    thumb_success = clear_thumbnails()
    print()
    
    # Summary
    print("=" * 60)
    if db_success and thumb_success:
        print("✓ ALL DATA CLEARED SUCCESSFULLY!")
        print("  System is now ready for fresh testing.")
    else:
        print("⚠ CLEARING COMPLETED WITH ERRORS")
        print("  Please check the messages above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
