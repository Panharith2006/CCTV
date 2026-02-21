"""
Reset Database Auto-Increment Counters
Clears all data and resets IDs to start from 1 for persons, features, and location_history tables
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mysql.connector
from config.mysql_config import MYSQL_CONFIG

def reset_database():
    """Reset all tables and auto-increment counters"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        print("="*80)
        print("RESET DATABASE AUTO-INCREMENT COUNTERS")
        print("="*80)
        
        # Get current counts
        cursor.execute("SELECT COUNT(*) FROM persons")
        person_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM features")
        feature_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM location_history")
        location_count = cursor.fetchone()[0]
        
        # Get current auto-increment values
        cursor.execute("SHOW TABLE STATUS LIKE 'persons'")
        persons_info = cursor.fetchone()
        persons_auto_inc = persons_info[10]  # Auto_increment column
        
        cursor.execute("SHOW TABLE STATUS LIKE 'features'")
        features_info = cursor.fetchone()
        features_auto_inc = features_info[10]
        
        cursor.execute("SHOW TABLE STATUS LIKE 'location_history'")
        location_info = cursor.fetchone()
        location_auto_inc = location_info[10]
        
        print("\nCurrent database state:")
        print(f"  persons:          {person_count} rows | Next ID: {persons_auto_inc}")
        print(f"  features:         {feature_count} rows | Next ID: {features_auto_inc}")
        print(f"  location_history: {location_count} rows | Next ID: {location_auto_inc}")
        
        if person_count == 0 and feature_count == 0 and location_count == 0:
            print("\n✓ All tables are already empty")
            print("  Resetting auto-increment counters only...")
            
            cursor.execute("ALTER TABLE location_history AUTO_INCREMENT = 1")
            cursor.execute("ALTER TABLE features AUTO_INCREMENT = 1")
            cursor.execute("ALTER TABLE persons AUTO_INCREMENT = 1")
            
            conn.commit()
            print("\n✓ Auto-increment counters reset to 1")
            
        else:
            print(f"\nThis will DELETE:")
            print(f"  {person_count} persons")
            print(f"  {feature_count} features")
            print(f"  {location_count} location history events")
            print(f"\nAnd reset all ID counters to start from 1")
            
            response = input("\n⚠️  Continue? (yes/no): ").strip().lower()
            
            if response != 'yes':
                print("\n❌ Operation cancelled")
                cursor.close()
                conn.close()
                return
            
            # Delete in correct order (foreign key constraints)
            print("\nDeleting data...")
            
            cursor.execute("DELETE FROM location_history")
            print(f"  ✓ Deleted {location_count} location history events")
            
            cursor.execute("DELETE FROM features")
            print(f"  ✓ Deleted {feature_count} features")
            
            cursor.execute("DELETE FROM persons")
            print(f"  ✓ Deleted {person_count} persons")
            
            # Reset auto-increment counters
            print("\nResetting auto-increment counters...")
            cursor.execute("ALTER TABLE location_history AUTO_INCREMENT = 1")
            print("  ✓ location_history: Next ID = 1")
            
            cursor.execute("ALTER TABLE features AUTO_INCREMENT = 1")
            print("  ✓ features: Next ID = 1")
            
            cursor.execute("ALTER TABLE persons AUTO_INCREMENT = 1")
            print("  ✓ persons: Next ID = 1")
            
            conn.commit()
        
        # Verify final state
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        cursor.execute("SELECT COUNT(*) FROM persons")
        print(f"  persons: {cursor.fetchone()[0]} rows")
        
        cursor.execute("SELECT COUNT(*) FROM features")
        print(f"  features: {cursor.fetchone()[0]} rows")
        
        cursor.execute("SELECT COUNT(*) FROM location_history")
        print(f"  location_history: {cursor.fetchone()[0]} rows")
        
        cursor.execute("SHOW TABLE STATUS LIKE 'persons'")
        persons_info = cursor.fetchone()
        print(f"\n  persons next ID: {persons_info[10]}")
        
        cursor.execute("SHOW TABLE STATUS LIKE 'features'")
        features_info = cursor.fetchone()
        print(f"  features next ID: {features_info[10]}")
        
        cursor.execute("SHOW TABLE STATUS LIKE 'location_history'")
        location_info = cursor.fetchone()
        print(f"  location_history next ID: {location_info[10]}")
        
        print("\n" + "="*80)
        print("✓ DATABASE RESET COMPLETED")
        print("="*80)
        print("\nAll tables are empty and IDs will start from 1")
        print("Next detected violator will be:")
        print("  - Person ID: 1")
        print("  - Feature ID: 1")
        print("  - Location History ID: 1")
        
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"\n❌ MySQL Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reset_database()
