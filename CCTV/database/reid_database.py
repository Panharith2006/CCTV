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
        #
        # FIELD MAPPING TO USER REQUIREMENTS:
        # - person_id          → person_id (int)
        # - first_seen         → first_seen_time (datetime)
        # - last_seen          → last_seen_time (datetime)
        # - violation_status   → status (enum: 'WARNING'/'ALERT')
        # - violation_reason   → violation_type ('MASK'/'HELMET'/'COMBINATION')
        # - last_camera_id     → camera_id (VARCHAR for flexibility: '1' or 'camera_front')
        # - thumbnail_path     → latest_snapshot_path (string)
        # Additional fields: appearance_count, status (tracking lifecycle), name (manual label),
        #                    is_reidentified (re-entry flag), detection_date (same-day filtering)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INT AUTO_INCREMENT PRIMARY KEY,          -- ✅ person_id (int)
                first_seen DATETIME,                               -- ✅ first_seen_time (datetime)
                last_seen DATETIME,                                -- ✅ last_seen_time (datetime)
                appearance_count INT DEFAULT 1,                    -- Additional: re-detection count
                status VARCHAR(20) DEFAULT 'active',               -- Additional: tracking lifecycle ('active'/'exited')
                name VARCHAR(255) DEFAULT NULL,                    -- Additional: manual label (for tools)
                thumbnail_path VARCHAR(512) DEFAULT NULL,          -- ✅ latest_snapshot_path (string)
                last_camera_id VARCHAR(100) DEFAULT NULL,          -- ✅ camera_id (VARCHAR not INT for flexibility)
                last_location VARCHAR(255) DEFAULT NULL,           -- Additional: human-readable location
                violation_status VARCHAR(20) DEFAULT NULL,         -- ✅ status (enum: 'WARNING'/'ALERT')
                violation_reason VARCHAR(255) DEFAULT NULL,        -- ✅ violation_type ('MASK'/'HELMET')
                is_reidentified TINYINT(1) DEFAULT 0,              -- Additional: re-entry flag
                detection_date DATE,                               -- Additional: same-day filtering
                INDEX idx_status (status),
                INDEX idx_violation (violation_status),
                INDEX idx_detection_date (detection_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Feature vectors table
        # ONE feature per person (updated on re-identification)
        # FIELD MAPPING:
        # - feature_vector → 128D feature vector (stored as JSON array)
        # - camera_id      → camera_id (last camera where feature was extracted)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id INT AUTO_INCREMENT PRIMARY KEY,         -- Unique feature ID
                person_id INT,                                     -- Foreign key to persons
                feature_vector TEXT,                               -- ✅ 128D feature vector (JSON)
                timestamp DATETIME,                                -- Last update time
                camera_id VARCHAR(100),                            -- ✅ camera_id (last capture location)
                is_masked TINYINT(1) DEFAULT 0,                    -- Context: masked state
                is_helmeted TINYINT(1) DEFAULT 0,                  -- Context: helmeted state
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_id (person_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Location history table (track movement across cameras)
        # FIELD MAPPING:
        # - camera_id → location_history (list[int]) - Sequential camera transitions
        # This table stores the full path of person movement: camera1 → camera2 → camera3
        # Event-based storage: only inserts when camera changes (not per-frame)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_history (
                history_id INT AUTO_INCREMENT PRIMARY KEY,         -- Unique entry ID
                person_id INT,                                     -- Foreign key to persons
                camera_id VARCHAR(100),                            -- ✅ camera_id (part of location_history list)
                location VARCHAR(255),                             -- Human-readable location
                timestamp DATETIME,                                -- Event time
                is_masked TINYINT(1) DEFAULT 0,                    -- Mask state at this event (tracks changes)
                is_helmeted TINYINT(1) DEFAULT 0,                  -- Helmet state at this event
                event_type VARCHAR(50) DEFAULT 'movement',         -- 'first_detection', 'movement', 'exit'
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
                INDEX idx_person_timestamp (person_id, timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Note: suspect_images and detections tables have been removed (redundant)
        # - suspect_images: Use thumbnail_path in persons table instead
        # - detections: Not needed for violation-only storage (creates millions of rows)
        
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
            name: Deprecated (not used in violation-only system)
            violation_status: 'WARNING' or 'ALERT'
            violation_reason: 'MASK', 'HELMET', 'ERRATIC_MOTION', 'LOITERING', 'COMBINATION'
            is_masked: Current mask status (for feature context)
            is_helmeted: Current helmet status (for feature context)
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        today = now.date()
        
        # Insert person with location and violation status (name column removed)
        cursor.execute("""
            INSERT INTO persons (
                first_seen, last_seen, thumbnail_path,
                last_camera_id, last_location,
                violation_status, violation_reason, detection_date, is_reidentified
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0)
        """, (now, now, thumbnail_path, camera_id, location, 
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
        Get all person features for matching (ONE feature per person)
        
        CHANGED: Single feature per person + same-day filtering for faster matching
        
        Args:
            same_day_only: If True, only return features from today's violations
        
        Returns:
            List of dicts with person_id, features, violation_status, violation_reason
            Note: Returns ONE feature per person (latest updated feature)
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
        Update person's last seen and REPLACE existing feature (single feature per person)
        
        CHANGED: Now UPDATES the existing feature instead of adding new ones.
        Strategy: One feature per person for simplicity and efficiency.
        
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
        
        # UPDATE existing feature (replace with latest)
        feature_json = json.dumps(feature_vector.tolist())
        cursor.execute("""
            UPDATE features
            SET feature_vector = %s, timestamp = %s, camera_id = %s, is_masked = %s, is_helmeted = %s
            WHERE person_id = %s
        """, (feature_json, now, camera_id, int(is_masked), int(is_helmeted), person_id))
        
        self.conn.commit()
    
    # Note: add_detection and get_last_detection removed
    # detections table was creating millions of rows and not used for violation tracking
    
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
    
    # Note: save_suspect_image removed
    # suspect_images table was redundant with thumbnail_path in persons table
    # Use thumbnail_path parameter in add_person() instead
    
    def mark_person_exit(self, person_id, camera_id, location):
        """
        Mark violation person as exited (final storage point)
        
        Only violation persons (those in DB) can be marked as exited
        Memory-only persons are simply deleted from memory
        
        Args:
            person_id: Person identifier (DB ID, not memory ID)
            camera_id: Camera where exit detected
            location: Location name
        
        Note: Prevents duplicate exit events by checking current status
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Check current status to prevent duplicate exit events
        cursor.execute("""
            SELECT status FROM persons WHERE person_id = %s
        """, (person_id,))
        result = cursor.fetchone()
        
        if not result:
            return  # Person doesn't exist
        
        current_status = result[0]
        if current_status == 'exited':
            # Already marked as exited, don't create duplicate exit event
            return
        
        # Update person status
        cursor.execute("""
            UPDATE persons
            SET status = 'exited', last_seen = %s
            WHERE person_id = %s
        """, (now, person_id))
        
        # Add exit event to location history (only if not already exited)
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
    
    def get_suspect_thumbnail(self, person_id):
        """
        Get thumbnail image for a person
        
        Args:
            person_id: Person identifier
        
        Returns:
            Thumbnail path or None
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT thumbnail_path
            FROM persons
            WHERE person_id = %s
        """, (person_id,))
        
        row = cursor.fetchone()
        return row[0] if row else None
    
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
        
        # Persons with thumbnails
        cursor.execute("SELECT COUNT(*) FROM persons WHERE thumbnail_path IS NOT NULL")
        persons_with_thumbnails = cursor.fetchone()[0]
        
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
            'persons_with_thumbnails': persons_with_thumbnails,
            'today_violations': today_violations
        }
    
    def close(self):
        """Close MySQL connection"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("[Database] MySQL connection closed")
