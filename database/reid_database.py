import mysql.connector
from mysql.connector import Error
import numpy as np
import json
from datetime import datetime, timedelta


class ReIDDatabase:
    def __init__(self, config=None):
        if config is None:
            try:
                from config.mysql_config import MYSQL_CONFIG
                config = MYSQL_CONFIG
            except ImportError:
                raise ValueError(
                    "MySQL config not found. Create config/mysql_config.py from "
                    "config/mysql_config.py.example and fill in your credentials."
                )
        
        self.config = config
        self.conn = None
        self.retention_days = 2
        self.connect()
        self.create_tables()
        self.purge_old_data(retention_days=self.retention_days)
        print(f"[Database] Connected to MySQL database: {config['database']}")
    
    def connect(self):
        try:
            # First, connect without database to check if it exists
            conn_temp = mysql.connector.connect(
                host=self.config['host'],
                port=self.config.get('port', 3306),
                user=self.config['user'],
                password=self.config['password'],
                charset=self.config.get('charset', 'utf8mb4'),
                auth_plugin=self.config.get('auth_plugin', 'mysql_native_password')
            )
            cursor_temp = conn_temp.cursor()
            
            # Create database if it doesn't exist
            db_name = self.config['database']
            cursor_temp.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor_temp.close()
            conn_temp.close()
            
            # Now connect with the database
            self.conn = mysql.connector.connect(
                host=self.config['host'],
                port=self.config.get('port', 3306),
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config.get('charset', 'utf8mb4'),
                autocommit=self.config.get('autocommit', False),
                auth_plugin=self.config.get('auth_plugin', 'mysql_native_password')
            )
            print(f"[Database] Connected to MySQL database: {db_name}")
        except Error as e:
            raise ConnectionError(f"[Database] MySQL connection failed: {e}")
    
    def create_tables(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Person table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INT AUTO_INCREMENT PRIMARY KEY,
                appearance_count INT DEFAULT 1,
                feature_count INT DEFAULT 1,
                centroid TEXT DEFAULT NULL,
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
        # Migration: remove old timestamp columns and ensure centroid/feature_count exist.
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'persons'
            """, (self.config['database'],))
            cols = {row[0] for row in cursor.fetchall()}
            if 'first_seen' in cols:
                cursor.execute("ALTER TABLE persons DROP COLUMN first_seen")
            if 'last_seen' in cols:
                cursor.execute("ALTER TABLE persons DROP COLUMN last_seen")
            if 'centroid' not in cols:
                cursor.execute("ALTER TABLE persons ADD COLUMN centroid TEXT DEFAULT NULL")
            if 'feature_count' not in cols:
                cursor.execute("ALTER TABLE persons ADD COLUMN feature_count INT DEFAULT 1")
            self.conn.commit()
        except Exception:
            # Non-fatal: if migration fails (permissions, read-only), continue; DB may be upgraded manually
            pass

    def purge_old_data(self, retention_days=2):
       
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=retention_days)

        cursor.execute("DELETE FROM detections WHERE timestamp < %s", (cutoff,))
        cursor.execute("DELETE FROM features WHERE timestamp < %s", (cutoff,))

        # Remove persons that no longer have any recent features/detections.
        cursor.execute("""
            DELETE p FROM persons p
            LEFT JOIN features f ON f.person_id = p.person_id
            LEFT JOIN detections d ON d.person_id = p.person_id
            WHERE f.person_id IS NULL AND d.person_id IS NULL
        """)

        self.conn.commit()
    
    def add_person(self, feature_vector, camera_id, thumbnail_path=None, name=None, is_masked=False, is_helmeted=False):
        """Add new person to database"""
        if not (is_masked or is_helmeted):
            print("[Database] Skipping person storage because no mask/helmet was confirmed")
            return None

        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Insert person with centroid stored on the persons row (canonical vector)
        centroid_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO persons (thumbnail_path, name, centroid, feature_count)
            VALUES (%s, %s, %s, %s)
        """, (thumbnail_path, name, centroid_json, 1))
        person_id = cursor.lastrowid

        # Insert feature into audit table (optional) for traceability
        feature_json = centroid_json
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id, is_masked, is_helmeted)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (person_id, feature_json, now, camera_id, int(is_masked), int(is_helmeted)))
        
        self.conn.commit()
        print(f"[Database] Added new person: ID={person_id} name={name} (masked={is_masked}, helmeted={is_helmeted})")
        self.purge_old_data(retention_days=self.retention_days)
        return person_id
    
    def get_all_features(self):
        """Get all person features for matching"""
        cursor = self.conn.cursor()
        # Prefer using the centroid stored on the persons table when available
        cursor.execute("""
            SELECT p.person_id, p.centroid
            FROM persons p
            WHERE p.status='active' AND p.centroid IS NOT NULL
        """)
        
        results = []
        for row in cursor.fetchall():
            person_id = row[0]
            try:
                feature_vector = np.array(json.loads(row[1]))
            except Exception:
                feature_vector = None
            if feature_vector is not None:
                results.append({
                    'person_id': person_id,
                    'features': feature_vector
                })
        
        return results
    
    def update_person(self, person_id, feature_vector, camera_id, is_masked=False, is_helmeted=False, alpha=0.15, max_feature_count=20):
        """Update person's last seen and merge new feature into canonical centroid using EMA.

        Args:
            person_id: existing person id
            feature_vector: numpy array
            camera_id: camera id string
            is_masked/is_helmeted: booleans
            alpha: EMA alpha (0-1). If None, centroid will be updated as simple mean using counts.
            max_feature_count: cap for count used in mean calculation
        """
        cursor = self.conn.cursor()
        now = datetime.now()

        if not (is_masked or is_helmeted):
            print(f"[Database] Skipping update for person {person_id} because no mask/helmet was confirmed")
            return

        # Fetch existing centroid and feature_count
        cursor.execute("""
            SELECT centroid, feature_count FROM persons WHERE person_id = %s
        """, (person_id,))
        row = cursor.fetchone()
        if row:
            centroid_json, fcount = row[0], row[1] or 0
            try:
                centroid = np.array(json.loads(centroid_json)) if centroid_json else None
            except Exception:
                centroid = None
        else:
            centroid = None
            fcount = 0

        new_feat = np.array(feature_vector)
        # Merge using EMA if centroid exists
        if centroid is None:
            merged = new_feat
            new_count = 1
        else:
            if alpha is None:
                # simple incremental mean with cap
                n = min(fcount, max_feature_count)
                merged = (centroid * n + new_feat) / float(n + 1)
                new_count = min(fcount + 1, max_feature_count)
            else:
                merged = alpha * new_feat + (1.0 - alpha) * centroid
                new_count = min(fcount + 1, max_feature_count)

        merged_json = json.dumps(merged.tolist())

        # Update persons row with new centroid and metadata
        cursor.execute("""
            UPDATE persons
            SET appearance_count = appearance_count + 1, centroid = %s, feature_count = %s
            WHERE person_id = %s
        """, (merged_json, int(new_count), person_id))

        # Insert feature into audit table (optional)
        feature_json = json.dumps(new_feat.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id, is_masked, is_helmeted)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (person_id, feature_json, now, camera_id, int(is_masked), int(is_helmeted)))

        self.conn.commit()
        self.purge_old_data(retention_days=self.retention_days)
    
    def add_detection(self, person_id, camera_id, bbox, track_id):
        """Log a detection"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (person_id, camera_id, bbox, timestamp, track_id)
            VALUES (%s, %s, %s, %s, %s)
        """, (person_id, camera_id, json.dumps(bbox), datetime.now(), track_id))
        self.conn.commit()
        self.purge_old_data(retention_days=self.retention_days)

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
            SELECT appearance_count, feature_count, name, thumbnail_path, status
            FROM persons
            WHERE person_id = %s
        """, (person_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'appearance_count': row[0],
            'feature_count': row[1],
            'name': row[2],
            'thumbnail_path': row[3],
            'status': row[4],
        }
    
    def get_all_persons(self):
        """Get all persons with metadata"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_id, appearance_count, feature_count, name, thumbnail_path, status
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
        """Delete a person (soft delete)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE persons SET status = 'deleted' WHERE person_id = %s
        """, (person_id,))
        self.conn.commit()
        print(f"[Database] Deleted person {person_id}")
    
    def close(self):
        """Close MySQL connection"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("[Database] MySQL connection closed")
