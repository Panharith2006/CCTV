import mysql.connector
from mysql.connector import Error
import numpy as np
import json
from datetime import datetime


class ReIDDatabase:
    def __init__(self, config=None):
        if config is None:
            try:
                from config.mysql_config import MYSQL_CONFIG
                config = MYSQL_CONFIG
            except ImportError:
                raise ValueError(
                )
        
        self.config = config
        self.conn = None
        self.connect()
        self.create_tables()
        print(f"[Database] Connected to MySQL database: {config['database']}")
    
    def connect(self):
        """Establish MySQL connection"""
        try:
            self.conn = mysql.connector.connect(
                host=self.config['host'],
                port=self.config.get('port', 3306),
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config.get('charset', 'utf8mb4'),
                autocommit=self.config.get('autocommit', False)
            )
        except Error as e:
            raise ConnectionError(f"[Database] MySQL connection failed: {e}")
    
    def create_tables(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Person table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INT AUTO_INCREMENT PRIMARY KEY,
                first_seen DATETIME,
                last_seen DATETIME,
                appearance_count INT DEFAULT 1,
                status VARCHAR(20) DEFAULT 'active',
                name VARCHAR(255) DEFAULT NULL,
                thumbnail_path VARCHAR(512) DEFAULT NULL,
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Feature vectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT,
                feature_vector TEXT,
                timestamp DATETIME,
                camera_id VARCHAR(100),
                is_masked TINYINT(1) DEFAULT 0,
                is_helmeted TINYINT(1) DEFAULT 0,
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_id (person_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Detections table (audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT,
                camera_id VARCHAR(100),
                bbox TEXT,
                timestamp DATETIME,
                track_id INT,
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_timestamp (person_id, timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        self.conn.commit()
        print("[Database] MySQL tables created/verified")
    
    def add_person(self, feature_vector, camera_id, thumbnail_path=None, name=None, is_masked=False, is_helmeted=False):
        """Add new person to database"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Insert person
        cursor.execute("""
            INSERT INTO persons (first_seen, last_seen, thumbnail_path, name)
            VALUES (%s, %s, %s, %s)
        """, (now, now, thumbnail_path, name))
        person_id = cursor.lastrowid
        
        # Insert feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id, is_masked, is_helmeted)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (person_id, feature_json, now, camera_id, int(is_masked), int(is_helmeted)))
        
        self.conn.commit()
        print(f"[Database] Added new person: ID={person_id} name={name} (masked={is_masked}, helmeted={is_helmeted})")
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
    
    def update_person(self, person_id, feature_vector, camera_id, is_masked=False, is_helmeted=False):
        """Update person's last seen and add new feature"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Update person
        cursor.execute("""
            UPDATE persons
            SET last_seen = %s, appearance_count = appearance_count + 1
            WHERE person_id = %s
        """, (now, person_id))
        
        # Add new feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id, is_masked, is_helmeted)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (person_id, feature_json, now, camera_id, int(is_masked), int(is_helmeted)))
        
        self.conn.commit()
    
    def add_detection(self, person_id, camera_id, bbox, track_id):
        """Log a detection"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (person_id, camera_id, bbox, timestamp, track_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (person_id, camera_id, json.dumps(bbox), datetime.now(), track_id))
        self.conn.commit()

    def get_last_detection(self, person_id, camera_id):
        """Return last detection row for a person on a camera.

        Returns dict with keys: 'bbox' (list) and 'timestamp' (datetime)
        or None if no prior detection.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT bbox, timestamp
            FROM detections
            WHERE person_id = %s AND camera_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (person_id, camera_id))
        row = cursor.fetchone()
        if not row:
            return None
        bbox_json, ts = row[0], row[1]
        try:
            bbox = json.loads(bbox_json)
        except Exception:
            bbox = bbox_json
        return {'bbox': bbox, 'timestamp': ts}
    
    def get_person_stats(self, person_id):
        """Get statistics for a person"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT first_seen, last_seen, appearance_count, name, thumbnail_path
            FROM persons
            WHERE person_id = %s
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
            UPDATE persons SET name = %s WHERE person_id = %s
        """, (name, person_id))
        self.conn.commit()
        print(f"[Database] Updated person {person_id} name to: {name}")
    
    def merge_persons(self, source_id, target_id):
        """Merge source_id into target_id"""
        cursor = self.conn.cursor()
        
        # Move all features from source to target
        cursor.execute("""
            UPDATE features SET person_id = %s WHERE person_id = %s
        """, (target_id, source_id))
        
        # Move all detections
        cursor.execute("""
            UPDATE detections SET person_id = %s WHERE person_id = %s
        """, (target_id, source_id))
        
        # Mark source as merged
        cursor.execute("""
            UPDATE persons SET status = 'merged' WHERE person_id = %s
        """, (source_id,))
        
        self.conn.commit()
        print(f"[Database] Merged person {source_id} into {target_id}")
    
    def delete_person(self, person_id):
        # Delete a person (soft delete)
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE persons SET status = 'deleted' WHERE person_id = %s
        """, (person_id,))
        self.conn.commit()
        print(f"[Database] Deleted person {person_id}")
    
    def close(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("[Database] MySQL connection closed")
