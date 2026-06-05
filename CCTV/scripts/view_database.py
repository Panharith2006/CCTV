"""
Simple database viewer for violation-only CCTV system
Shows current contents of persons, features, and location_history tables
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.reid_database import ReIDDatabase

def print_database():
    """Print current database contents"""
    db = ReIDDatabase()
    cursor = db.conn.cursor(dictionary=True)
    
    print("="*80)
    print("DATABASE CONTENTS - VIOLATION-ONLY CCTV SYSTEM")
    print("="*80)
    
    # Persons table
    print("\n[1] PERSONS TABLE")
    print("-" * 80)
    cursor.execute("SELECT COUNT(*) as count FROM persons")
    person_count = cursor.fetchone()['count']
    print(f"Total persons: {person_count}")
    
    if person_count > 0:
        cursor.execute("""
            SELECT person_id, last_camera_id, first_seen, last_seen, 
                   appearance_count, violation_status, violation_reason, 
                   thumbnail_path
            FROM persons 
            ORDER BY person_id DESC 
            LIMIT 10
        """)
        print("\nRecent persons (limit 10):")
        for row in cursor:
            status = row['violation_status'] or 'N/A'
            reason = row['violation_reason'] or 'N/A'
            print(f"  P{row['person_id']:03d} | Camera {row['last_camera_id']} | "
                  f"{status:8s} | {reason:20s} | "
                  f"Seen {row['appearance_count']}x | "
                  f"First: {row['first_seen']} | Last: {row['last_seen']}")
    
    # Features table
    print("\n[2] FEATURES TABLE")
    print("-" * 80)
    cursor.execute("SELECT COUNT(*) as count FROM features")
    feature_count = cursor.fetchone()['count']
    print(f"Total features: {feature_count}")
    
    if feature_count > 0:
        cursor.execute("""
            SELECT person_id, LENGTH(feature_vector) as feature_size, timestamp
            FROM features 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        print("\nRecent features (limit 10):")
        for row in cursor:
            print(f"  P{row['person_id']:03d} | Feature size: {row['feature_size']} bytes | "
                  f"Created: {row['timestamp']}")
    
    # Location history table
    print("\n[3] LOCATION_HISTORY TABLE")
    print("-" * 80)
    cursor.execute("SELECT COUNT(*) as count FROM location_history")
    location_count = cursor.fetchone()['count']
    print(f"Total location events: {location_count}")
    
    if location_count > 0:
        cursor.execute("""
            SELECT person_id, camera_id, event_type, timestamp
            FROM location_history 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        print("\nRecent location events (limit 10):")
        for row in cursor:
            print(f"  P{row['person_id']:03d} | Camera {row['camera_id']} | "
                  f"{row['event_type']:8s} | {row['timestamp']}")
    
    # Data consistency check
    print("\n" + "="*80)
    print("DATA CONSISTENCY CHECK")
    print("="*80)
    
    # Check persons without features
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM persons 
        WHERE person_id NOT IN (SELECT DISTINCT person_id FROM features)
    """)
    missing_features = cursor.fetchone()['count']
    
    if missing_features > 0:
        print(f"⚠️  Warning: {missing_features} persons have no feature vectors!")
        cursor.execute("""
            SELECT person_id, last_camera_id, first_seen, violation_reason
            FROM persons 
            WHERE person_id NOT IN (SELECT DISTINCT person_id FROM features)
        """)
        print("  Persons without features:")
        for row in cursor:
            print(f"    P{row['person_id']:03d} | Camera {row['last_camera_id']} | "
                  f"{row['violation_reason']} | First seen: {row['first_seen']}")
    else:
        print("✓ All persons have corresponding feature vectors")
    
    # Check features without persons
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM features 
        WHERE person_id NOT IN (SELECT person_id FROM persons)
    """)
    orphan_features = cursor.fetchone()['count']
    
    if orphan_features > 0:
        print(f"⚠️  Warning: {orphan_features} feature vectors have no corresponding person!")
    else:
        print("✓ All features have corresponding persons")
    
    print("\n" + "="*80)
    
    cursor.close()
    db.conn.close()

if __name__ == "__main__":
    print_database()
