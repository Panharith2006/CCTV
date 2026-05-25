"""
Clear all persons from database for fresh testing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.reid_database import ReIDDatabase
from config.mysql_config import MYSQL_CONFIG

def clear_database():
    """Delete all persons, features, and location history"""
    db = ReIDDatabase(MYSQL_CONFIG)
    
    print("=" * 80)
    print("CLEAR ALL PERSONS FROM DATABASE")
    print("=" * 80)
    
    # Show current counts
    cursor = db.conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as count FROM persons")
    person_count = cursor.fetchone()['count']
    cursor.execute("SELECT COUNT(*) as count FROM features")
    feature_count = cursor.fetchone()['count']
    cursor.execute("SELECT COUNT(*) as count FROM location_history")
    history_count = cursor.fetchone()['count']
    
    print(f"\nCurrent database:")
    print(f"  Persons: {person_count}")
    print(f"  Features: {feature_count}")
    print(f"  Location events: {history_count}")
    
    if person_count == 0:
        print("\n✓ Database already empty!")
        db.close()
        return
    
    # Confirm deletion
    print(f"\n⚠️  This will DELETE all {person_count} persons and reset IDs to start from 1")
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Cancelled.")
        db.close()
        return
    
    # Delete all data
    try:
        cursor.execute("DELETE FROM location_history")
        deleted_history = cursor.rowcount
        
        cursor.execute("DELETE FROM features")
        deleted_features = cursor.rowcount
        
        cursor.execute("DELETE FROM persons")
        deleted_persons = cursor.rowcount
        
        # Reset auto increment
        cursor.execute("ALTER TABLE persons AUTO_INCREMENT = 1")
        cursor.execute("ALTER TABLE features AUTO_INCREMENT = 1")
        cursor.execute("ALTER TABLE location_history AUTO_INCREMENT = 1")
        
        db.conn.commit()
        
        print("\n✓ Database cleared successfully!")
        print(f"  Deleted {deleted_persons} persons")
        print(f"  Deleted {deleted_features} features")
        print(f"  Deleted {deleted_history} location events")
        print(f"  Reset all auto-increment counters to 1")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        db.conn.rollback()
    
    finally:
        cursor.close()
        db.close()

if __name__ == "__main__":
    clear_database()
