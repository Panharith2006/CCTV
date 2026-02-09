import mysql.connector
from mysql.connector import Error
import numpy as np
import json
from datetime import datetime


class ReIDDatabase:
    def __init__(self, config=None):
        """
        Initialize MySQL connection for ReID database
        
        Args:
            config: dict with keys: host, port, user, password, database
                   If None, tries to import from config/mysql_config.py
        """
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
        self.connect()
        self.create_tables()
        print(f"[Database] Connected to MySQL database: {config['database']}")
    
    def connect(self):
        """Establish MySQL connection"""
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
        
        # Person table with location and violation tracking
        # REDESIGNED: Single violation_status and violation_reason fields
        # Only persons with violations are saved (violation-only storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INT AUTO_INCREMENT PRIMARY KEY,
                first_seen DATETIME,
                last_seen DATETIME,
                appearance_count INT DEFAULT 1,
                status VARCHAR(20) DEFAULT 'active',
                name VARCHAR(255) DEFAULT NULL,
                thumbnail_path VARCHAR(512) DEFAULT NULL,
                last_camera_id VARCHAR(100) DEFAULT NULL,
                last_location VARCHAR(255) DEFAULT NULL,
                violation_status VARCHAR(20) DEFAULT NULL,
                violation_reason VARCHAR(255) DEFAULT NULL,
                is_reidentified TINYINT(1) DEFAULT 0,
                detection_date DATE,
                INDEX idx_status (status),
                INDEX idx_violation (violation_status),
                INDEX idx_detection_date (detection_date)
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
        
        # Location history table (track movement across cameras)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_history (
                history_id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT,
                camera_id VARCHAR(100),
                location VARCHAR(255),
                timestamp DATETIME,
                is_masked TINYINT(1) DEFAULT 0,
                is_helmeted TINYINT(1) DEFAULT 0,
                event_type VARCHAR(50) DEFAULT 'movement',
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_timestamp (person_id, timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Suspect images table (store images for suspects only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suspect_images (
                image_id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT,
                image_path VARCHAR(1024),
                timestamp DATETIME,
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_id (person_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        self.conn.commit()
        print("[Database] MySQL tables created/verified")
    
    def add_person(self, feature_vector, camera_id, location=None, thumbnail_path=None, name=None, violation_status=None, violation_reason=None, is_masked=False, is_helmeted=False):
        """
        Add new person to database (ONLY when violations detected)
        
        CRITICAL: This should ONLY be called for persons WITH violations
        Normal compliant persons are tracked in memory only
        
        Args:
            feature_vector: 128D ReID feature vector
            camera_id: Camera identifier
            location: Camera location
            thumbnail_path: Path to saved image
            name: Person name (if enrolled)
            violation_status: 'WARNING' or 'ALERT'
            violation_reason: 'MASK', 'HELMET', 'ERRATIC_MOTION', 'LOITERING', 'COMBINATION'
            is_masked: Current mask status (for feature context)
            is_helmeted: Current helmet status (for feature context)
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        today = now.date()
        
        # Insert person with location and violation status
        cursor.execute("""
            INSERT INTO persons (
                first_seen, last_seen, thumbnail_path, name, 
                last_camera_id, last_location,
                violation_status, violation_reason, detection_date, is_reidentified
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
        """, (now, now, thumbnail_path, name, camera_id, location, 
               violation_status, violation_reason, today))
        person_id = cursor.lastrowid
        
        # Insert feature
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            INSERT INTO features (person_id, feature_vector, timestamp, camera_id, is_masked, is_helmeted)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (person_id, feature_json, now, camera_id, int(is_masked), int(is_helmeted)))
        
        # Add initial location history
        cursor.execute("""
            INSERT INTO location_history (person_id, camera_id, location, timestamp, is_masked, is_helmeted, event_type)
            VALUES (%s, %s, %s, %s, %s, %s, 'first_detection')
        """, (person_id, camera_id, location, now, int(is_masked), int(is_helmeted)))
        
        self.conn.commit()
        print(f"[Database] ✅ NEW VIOLATION: ID={person_id} | Status={violation_status} | Reason={violation_reason} | Location={location}")
        return person_id
    
    def get_all_features(self, same_day_only=True):
        """
        Get all person features for matching
        
        CHANGED: Now supports same-day filtering for faster, more accurate matching
        
        Args:
            same_day_only: If True, only return features from today's violations
        
        Returns:
            List of dicts with person_id, features, violation_status, violation_reason
        """
        cursor = self.conn.cursor()
        
        if same_day_only:
            today = datetime.now().date()
            cursor.execute("""
                SELECT p.person_id, f.feature_vector, p.violation_status, p.violation_reason
                FROM features f
                JOIN persons p ON f.person_id = p.person_id
                WHERE p.status IN ('active', 'exited') 
                  AND p.detection_date = %s
            """, (today,))
        else:
            cursor.execute("""
                SELECT p.person_id, f.feature_vector, p.violation_status, p.violation_reason
                FROM features f
                JOIN persons p ON f.person_id = p.person_id
                WHERE p.status IN ('active', 'exited')
            """)
        
        results = []
        for row in cursor.fetchall():
            person_id = row[0]
            feature_vector = np.array(json.loads(row[1]))
            violation_status = row[2]
            violation_reason = row[3]
            results.append({
                'person_id': person_id,
                'features': feature_vector,
                'violation_status': violation_status,
                'violation_reason': violation_reason
            })
        
        return results
    
    def update_person(self, person_id, feature_vector, camera_id, is_masked=False, is_helmeted=False, mark_reidentified=True):
        """
        Update person's last seen and add new feature
        
        Args:
            mark_reidentified: If True, marks person as re-identified (not new suspect)
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Update person - mark as reidentified if this is a re-detection
        if mark_reidentified:
            cursor.execute("""
                UPDATE persons
                SET last_seen = %s, appearance_count = appearance_count + 1, is_reidentified = 1
                WHERE person_id = %s
            """, (now, person_id))
        else:
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
        """Delete a person (soft delete)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE persons SET status = 'deleted' WHERE person_id = %s
        """, (person_id,))
        self.conn.commit()
        print(f"[Database] Deleted person {person_id}")
    
    def update_person_location(self, person_id, camera_id, location, is_masked=False, is_helmeted=False):
        """
        Update person's location when they move to a new camera
        (Track movement across cameras - violation persons only)
        
        ✅ VERIFICATION POINT 3: Event-Based Storage (NOT per-frame)
        - Only stores when camera changes (meaningful event)
        - Does NOT store every frame update
        - Maintains "event-based DB" principle
        - Avoids database noise and unnecessary writes
        
        Args:
            person_id: Person identifier
            camera_id: New camera identifier
            location: Camera location (e.g., "Entrance", "Hallway")
            is_masked: Current mask status
            is_helmeted: Current helmet status
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Update person's last seen and location
        cursor.execute("""
            UPDATE persons
            SET last_seen = %s, last_camera_id = %s, last_location = %s
            WHERE person_id = %s
        """, (now, camera_id, location, person_id))
        
        # ✅ Check if camera changed (don't add duplicate location entries)
        cursor.execute("""
            SELECT camera_id FROM location_history
            WHERE person_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (person_id,))
        
        last_camera = cursor.fetchone()
        
        # ✅ Only add new location entry if camera changed (EVENT-BASED, not per-frame)
        if not last_camera or last_camera[0] != camera_id:
            cursor.execute("""
                INSERT INTO location_history (person_id, camera_id, location, timestamp, is_masked, is_helmeted, event_type)
                VALUES (%s, %s, %s, %s, %s, %s, 'movement')
            """, (person_id, camera_id, location, now, int(is_masked), int(is_helmeted)))
            print(f"[Database] Person {person_id} moved to {location} (Camera: {camera_id})")
        
        self.conn.commit()
    
    def save_suspect_image(self, person_id, image_path, timestamp=None):
        """
        Save image for violation person (called during violation creation)
        
        Args:
            person_id: Person identifier
            image_path: Path to saved image
            timestamp: When image was captured
        """
        cursor = self.conn.cursor()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        cursor.execute("""
            INSERT INTO suspect_images (person_id, image_path, timestamp)
            VALUES (%s, %s, %s)
        """, (person_id, image_path, timestamp))
        
        self.conn.commit()
        print(f"[Database] Violation image saved for Person {person_id}: {image_path}")
    
    def mark_person_exit(self, person_id, camera_id, location):
        """
        Mark violation person as exited (final storage point)
        
        Only violation persons (those in DB) can be marked as exited
        Memory-only persons are simply deleted from memory
        
        Args:
            person_id: Person identifier (DB ID, not memory ID)
            camera_id: Camera where exit detected
            location: Location name
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Update person status
        cursor.execute("""
            UPDATE persons
            SET status = 'exited', last_seen = %s
            WHERE person_id = %s
        """, (now, person_id))
        
        # Add exit event to location history
        cursor.execute("""
            INSERT INTO location_history (person_id, camera_id, location, timestamp, event_type)
            VALUES (%s, %s, %s, %s, 'exit')
        """, (person_id, camera_id, location, now))
        
        self.conn.commit()
        print(f"[Database] 🚪 Person {person_id} EXITED scene at {location}")
    
    def get_person_location_history(self, person_id):
        """
        Get movement history across cameras
        
        Args:
            person_id: Person identifier
        
        Returns:
            List of location records
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT camera_id, location, timestamp, is_masked, is_helmeted, event_type
            FROM location_history
            WHERE person_id = %s
            ORDER BY timestamp ASC
        """, (person_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'camera_id': row[0],
                'location': row[1],
                'timestamp': row[2],
                'is_masked': bool(row[3]),
                'is_helmeted': bool(row[4]),
                'event_type': row[5]
            })
        
        return results
    
    def get_suspect_images(self, person_id):
        """
        Get all suspect images for a person
        
        Args:
            person_id: Person identifier
        
        Returns:
            List of image records
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT image_path, timestamp
            FROM suspect_images
            WHERE person_id = %s
            ORDER BY timestamp DESC
        """, (person_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'image_path': row[0],
                'timestamp': row[1]
            })
        
        return results
    
    def get_summary(self):
        """
        Get database statistics (violation-only storage)
        
        Returns:
            Dictionary with database statistics
        """
        cursor = self.conn.cursor()
        
        # Total violations (all persons in DB are violations)
        cursor.execute("SELECT COUNT(*) FROM persons")
        total_violations = cursor.fetchone()[0]
        
        # Active violations
        cursor.execute("SELECT COUNT(*) FROM persons WHERE status = 'active'")
        active_violations = cursor.fetchone()[0]
        
        # Exited violations
        cursor.execute("SELECT COUNT(*) FROM persons WHERE status = 'exited'")
        exited = cursor.fetchone()[0]
        
        # Warnings vs Alerts
        cursor.execute("SELECT COUNT(*) FROM persons WHERE violation_status = 'WARNING'")
        warnings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM persons WHERE violation_status = 'ALERT'")
        alerts = cursor.fetchone()[0]
        
        # Re-identified suspects
        cursor.execute("SELECT COUNT(*) FROM persons WHERE is_reidentified = 1")
        reidentified = cursor.fetchone()[0]
        
        # Total suspect images
        cursor.execute("SELECT COUNT(*) FROM suspect_images")
        total_suspect_images = cursor.fetchone()[0]
        
        # Today's violations
        today = datetime.now().date()
        cursor.execute("SELECT COUNT(*) FROM persons WHERE detection_date = %s", (today,))
        today_violations = cursor.fetchone()[0]
        
        return {
            'total_violations': total_violations,
            'active_violations': active_violations,
            'exited': exited,
            'warnings': warnings,
            'alerts': alerts,
            'reidentified': reidentified,
            'total_suspect_images': total_suspect_images,
            'today_violations': today_violations
        }
    
    def close(self):
        """Close MySQL connection"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("[Database] MySQL connection closed")
