import sqlite3
import numpy as np
import json
from datetime import datetime


class ReIDDatabase:
    def __init__(self, db_path="reid_database.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        print(f"[Database] Connected to {db_path}")
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Person table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen DATETIME,
                last_seen DATETIME,
                appearance_count INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                name TEXT DEFAULT NULL,
                thumbnail_path TEXT DEFAULT NULL
            )
        """)
        
        # Migrate existing database: add new columns if they don't exist
        try:
            cursor.execute("SELECT name FROM persons LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE persons ADD COLUMN name TEXT DEFAULT NULL")
            print("[Database] Migrated: added 'name' column")
        
        try:
            cursor.execute("SELECT thumbnail_path FROM persons LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE persons ADD COLUMN thumbnail_path TEXT DEFAULT NULL")
            print("[Database] Migrated: added 'thumbnail_path' column")
        
        # Feature vectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                feature_vector TEXT,
                timestamp DATETIME,
                camera_id TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)
        
        # Detections table (for audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                camera_id TEXT,
                bbox TEXT,
                timestamp DATETIME,
                track_id INTEGER,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        """)
        
        self.conn.commit()
        print("[Database] Tables created")
    
    def add_person(self, feature_vector, camera_id, thumbnail_path=None, name=None):
        """Add new person to database"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Insert person
        cursor.execute("""
            INSERT INTO persons (first_seen, last_seen, thumbnail_path, name)
            VALUES (?, ?, ?, ?)
        """, (now, now, thumbnail_path, name))
        person_id = cursor.lastrowid
        
        # Insert feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id)
            VALUES (?, ?, ?, ?)
        """, (person_id, feature_json, now, camera_id))
        
        self.conn.commit()
        print(f"[Database] Added new person: ID={person_id} name={name}")
        return person_id
    
    def get_all_features(self):
        """Get all person features for matching"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_id, feature_vector
            FROM features
            WHERE person_id IN (SELECT person_id FROM persons WHERE status='active')
        """)
        
        results = []
        for row in cursor.fetchall():
            person_id = row[0]
            feature_vector = np.array(json.loads(row[1]))
            results.append({
                'person_id': person_id,
                'features': feature_vector
            })
        
        return results
    
    def update_person(self, person_id, feature_vector, camera_id):
        """Update person's last seen and add new feature"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Update person
        cursor.execute("""
            UPDATE persons
            SET last_seen = ?, appearance_count = appearance_count + 1
            WHERE person_id = ?
        """, (now, person_id))
        
        # Add new feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id)
            VALUES (?, ?, ?, ?)
        """, (person_id, feature_json, now, camera_id))
        
        self.conn.commit()
    
    def add_detection(self, person_id, camera_id, bbox, track_id):
        """Log a detection"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (person_id, camera_id, bbox, timestamp, track_id)
            VALUES (?, ?, ?, ?, ?)
        """, (person_id, camera_id, json.dumps(bbox), datetime.now(), track_id))
        self.conn.commit()
    
    def get_person_stats(self, person_id):
        """Get statistics for a person"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT first_seen, last_seen, appearance_count, name, thumbnail_path
            FROM persons
            WHERE person_id = ?
        """, (person_id,))
        return cursor.fetchone()
    
    def get_all_persons(self):
        """Get all persons with metadata"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_id, first_seen, last_seen, appearance_count, name, thumbnail_path
            FROM persons
            WHERE status='active'
            ORDER BY person_id
        """)
        return cursor.fetchall()
    
    def update_person_name(self, person_id, name):
        """Update person's name"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE persons SET name = ? WHERE person_id = ?
        """, (name, person_id))
        self.conn.commit()
        print(f"[Database] Updated person {person_id} name to: {name}")
    
    def merge_persons(self, source_id, target_id):
        """Merge source_id into target_id"""
        cursor = self.conn.cursor()
        
        # Move all features from source to target
        cursor.execute("""
            UPDATE features SET person_id = ? WHERE person_id = ?
        """, (target_id, source_id))
        
        # Move all detections
        cursor.execute("""
            UPDATE detections SET person_id = ? WHERE person_id = ?
        """, (target_id, source_id))
        
        # Mark source as merged
        cursor.execute("""
            UPDATE persons SET status = 'merged' WHERE person_id = ?
        """, (source_id,))
        
        self.conn.commit()
        print(f"[Database] Merged person {source_id} into {target_id}")
    
    def delete_person(self, person_id):
        """Delete a person (soft delete)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE persons SET status = 'deleted' WHERE person_id = ?
        """, (person_id,))
        self.conn.commit()
        print(f"[Database] Deleted person {person_id}")
    
    def close(self):
        self.conn.close()
