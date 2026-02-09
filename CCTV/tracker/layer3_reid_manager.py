import numpy as np
import cv2
import os
import time
from detector.layer2_reid_extractor import ReIDExtractor


class ReIDManager:
    """
    Re-Identification Manager - REVISED FOR THEFT/SUSPICIOUS BEHAVIOR DETECTION
    
    CRITICAL CHANGES:
    1. Purpose: Detect theft-related suspicious behavior (mask/helmet = identity concealment)
    2. Features: 128D body-based ReID (not face recognition)
    3. ID Assignment: Permanent IDs ONLY for violations (normal persons get temporary IDs)
    4. Matching Scope: Same-day violations only (faster, more accurate)
    5. Storage: Violation-only (normal persons never saved to database)
    
    Key Principles:
    - Mask/helmet inside buildings = ABNORMAL (suspicious, not safety compliance)
    - ONE ID per person (consistent even if violation status changes)
    - Normal people: memory-only tracking (M1, M2, ...)
    - Violation people: database storage with permanent IDs
    """
    def __init__(self, db_config=None, similarity_threshold=0.7, thumbnail_dir="thumbnails", camera_location=None, debug_level=1):
        # Try MySQL first, fall back to in-memory database
        try:
            from database.reid_database import ReIDDatabase
            self.db = ReIDDatabase(db_config)
        except (ImportError, ConnectionError, Exception) as e:
            print(f"[ReIDManager] MySQL not available ({e}), using in-memory database")
            from database.reid_database_memory import ReIDDatabase as MemoryDB
            self.db = MemoryDB(db_config)
        
        self.reid_extractor = ReIDExtractor()
        self.similarity_threshold = similarity_threshold
        self.thumbnail_dir = thumbnail_dir
        self.camera_location = camera_location or "Unknown Location"
        self.debug_level = debug_level  # 0=silent, 1=summary, 2=events, 3=verbose
        
        # Create thumbnail directories
        os.makedirs(thumbnail_dir, exist_ok=True)
        suspect_dir = os.path.join(thumbnail_dir, "suspects")
        os.makedirs(suspect_dir, exist_ok=True)
        
        # Tracking state
        self.active_tracks = {}  # track_id -> {'person_id': int, 'last_seen': timestamp, 'frames_tracked': int}
        self.track_exit_timeout = 30  # seconds before considering track exited
        
        # In-memory tracking for persons WITHOUT violations (not saved to DB)
        self.memory_persons = {}  # temp_id -> {'features': vector, 'first_seen': time, 'last_seen': time}
        self.next_temp_id = 1
        self.memory_cleanup_interval = 30  # seconds

        if self.debug_level >= 1:
            print(f"[ReIDManager] Initialized with threshold={similarity_threshold}")
            print(f"[ReIDManager] Camera location: {self.camera_location}")
            print(f"[ReIDManager] [!] THEFT DETECTION MODE: Mask/Helmet = Identity Concealment")
            print(f"[ReIDManager] Storage strategy: VIOLATION-ONLY (normal persons not saved)")
            print(f"[ReIDManager]    - WITH mask = WARNING (suspicious)")
            print(f"[ReIDManager]    - WITH helmet = ALERT (suspicious)")
            print(f"[ReIDManager]    - No violations = Normal (memory-only, not saved)")
            print(f"[ReIDManager] Matching scope: Same-day violations only")
    
    def save_thumbnail(self, frame, bbox, person_id, is_suspect=False):
        """
        Save person crop as thumbnail
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            person_id: Person identifier
            is_suspect: If True, saves to suspect folder
        
        Returns:
            filepath: Path to saved thumbnail
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            # Different folder for suspects
            if is_suspect:
                suspect_dir = os.path.join(self.thumbnail_dir, "suspects")
                os.makedirs(suspect_dir, exist_ok=True)
                filename = f"suspect_{person_id:03d}_{int(time.time()*1000)}.jpg"
                filepath = os.path.join(suspect_dir, filename)
            else:
                filename = f"person_{person_id:03d}_{int(time.time()*1000)}.jpg"
                filepath = os.path.join(self.thumbnail_dir, filename)
            
            cv2.imwrite(filepath, crop)
            return filepath
        except Exception as e:
            print(f"[ReIDManager] Failed to save thumbnail: {e}")
            return None

    def _calculate_bbox_distance(self, bbox1, bbox2):
        """
        Calculate Euclidean distance between centers of two bounding boxes.
        Used for spatial validation - two people can't be in same place at same time.
        
        Args:
            bbox1, bbox2: Bounding boxes in format [x1, y1, x2, y2]
        
        Returns:
            float: Distance in pixels between bbox centers
        """
        # Calculate centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        return distance

    def identify_person(self, frame, bbox, camera_id, is_masked=False, is_helmeted=False, assigned_person_ids=None):
        """
        Identify person using 128D body features (NOT face recognition)
        
        REVISED DETECTION LOGIC:
        1. Mask/Helmet detection FIRST (check for violations)
        2. Extract 128D ReID features from body (FOR ALL PERSONS - ALWAYS)
        3. Compare ONLY with same-day violations (database) + memory (normal persons)
        4. Save to DB ONLY if violations exist
        5. Permanent ID assigned ONLY when violation detected
        
        ✅ VERIFICATION POINT 1: Feature Extraction for ALL Persons
        - Normal persons: Features extracted → Compared → NOT saved
        - Violation persons: Features extracted → Compared → Saved to DB
        - This enables "suspect removed mask later" re-identification
        
        ✅ VERIFICATION POINT 2: is_reidentified Definition (CLARIFIED)
        - TRUE = Person was in database from PREVIOUS tracking session, now returned
        - FALSE = New person OR continuous tracking of existing person
        - Check: Person in DB + NOT in active_tracks = TRUE re-identification
        
        CRITICAL: Normal persons get temporary memory IDs (M1, M2...)
                  Violation persons get permanent database IDs (1, 2, 3...)
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            camera_id: Camera identifier
            is_masked: True = wearing mask (SUSPICIOUS - identity concealment)
            is_helmeted: True = wearing helmet (SUSPICIOUS - identity concealment)
            assigned_person_ids: Set of IDs already assigned in this frame (prevents collision)
        
        Returns:
            tuple: (person_id, is_reidentified)
                person_id: DB ID (int) for violations, or memory ID (str "M#") for normal
                is_reidentified: True if returning suspect, False if new/continuous
        """
        # STEP 1: Extract 128D feature vector from body (NOT face)
        # ✅ IMPORTANT: This happens for ALL persons (normal + violation)
        features = self.reid_extractor.extract_features(frame, bbox)
        
        if features is None:
            if self.debug_level >= 3:
                print("[ReIDManager] [X] Failed to extract 128D features")
            return None
        
        # STEP 2: Check for violations FIRST (before any DB operations)
        has_violation = is_masked or is_helmeted
        
        # Determine violation type and status
        if has_violation:
            violation_reasons = []
            violation_status = None
            
            if is_helmeted:
                violation_reasons.append("HELMET")
                violation_status = "ALERT"  # Helmet = ALERT (higher priority)
            
            if is_masked:
                violation_reasons.append("MASK")
                if not violation_status:  # Only set if not already ALERT
                    violation_status = "WARNING"  # Mask = WARNING
            
            violation_reason = "+".join(violation_reasons) if len(violation_reasons) > 1 else violation_reasons[0]
            
            if self.debug_level >= 2:
                print(f"[ReIDManager] [!] VIOLATION DETECTED: {violation_reason} | Status: {violation_status}")
        
        # STEP 3: Compare with database (same-day violations only) + memory (normal persons)
        # ✅ ALL persons are compared (even if no violation) for re-identification capability
        all_db_features = self.db.get_all_features(same_day_only=True)
        best_match_id = None
        best_similarity = 0.0
        match_source = None  # 'database' or 'memory'
        
        # Compare with database (saved violations from today)
        for person_data in all_db_features:
            pid = person_data['person_id']
            stored_features = person_data['features']
            similarity = self.reid_extractor.compare_features(features, stored_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = pid
                match_source = 'database'
        
        # Also compare with memory (normal persons not in DB)
        for temp_id, person_data in self.memory_persons.items():
            stored_features = person_data['features']
            similarity = self.reid_extractor.compare_features(features, stored_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = temp_id
                match_source = 'memory'
        
        # STEP 4: Decision based on match and violation status
        # ✅ COLLISION DETECTION: Check if matched ID is already assigned to another track
        # ✅ SPATIAL VALIDATION: Two people can't be in the same place at the same time
        if best_similarity >= self.similarity_threshold:
            if assigned_person_ids is not None and best_match_id in assigned_person_ids:
                # COLLISION DETECTED! This ID is already assigned to another active track
                # Additional validation: check spatial distance
                # Find the existing track with this person_id and check distance
                collision_confirmed = False
                for track_id, track_data in self.active_tracks.items():
                    if track_data.get('person_id') == best_match_id:
                        # Found existing track with same person_id
                        existing_bbox = track_data.get('bbox')
                        if existing_bbox is not None:
                            spatial_distance = self._calculate_bbox_distance(bbox, existing_bbox)
                            # If distance > 200 pixels, these are definitely different people
                            # (same person can only move ~50-100 pixels between frames)
                            if spatial_distance > 200:
                                collision_confirmed = True
                                if self.debug_level >= 2:
                                    print(f"[ReIDManager] [X] FALSE POSITIVE MATCH: ID={best_match_id} already active {spatial_distance:.0f}px away | Sim={best_similarity:.2f} | These are DIFFERENT people")
                                break
                        else:
                            # No bbox stored, assume collision
                            collision_confirmed = True
                            break
                
                if collision_confirmed:
                    # Reject this match - treat as new person
                    if self.debug_level >= 2:
                        print(f"[ReIDManager] [X] COLLISION PREVENTED: ID={best_match_id} already assigned | Sim={best_similarity:.2f} | Treating as new person")
                    best_similarity = 0.0
                    best_match_id = None
        
        if best_similarity >= self.similarity_threshold:
            # MATCHED to existing person
            if match_source == 'database':
                # Person exists in DB (has/had violations) - UPDATE
                
                # ✅ VERIFICATION POINT 2: Determine TRUE re-identification
                # Check if this person is currently being actively tracked
                is_currently_tracked = any(
                    track_data['person_id'] == best_match_id 
                    for track_data in self.active_tracks.values()
                )
                
                # TRUE re-identification = Person in DB but NOT currently tracked
                # (They left and came back)
                is_reidentified = not is_currently_tracked
                
                self.db.update_person(
                    person_id=best_match_id,
                    feature_vector=features,
                    camera_id=camera_id,
                    is_masked=is_masked,
                    is_helmeted=is_helmeted,
                    mark_reidentified=is_reidentified
                )
                self.db.update_person_location(
                    person_id=best_match_id,
                    camera_id=camera_id,
                    location=self.camera_location,
                    is_masked=is_masked,
                    is_helmeted=is_helmeted
                )
                
                if is_reidentified:
                    status_msg = "🔄 RE-IDENTIFIED (RETURNED)"
                else:
                    status_msg = "✅ MATCHED (CONTINUOUS)"
                
                if self.debug_level >= 3:
                    print(f"[ReIDManager] {status_msg}: Person ID={best_match_id} | Sim={best_similarity:.2f}")
                return best_match_id, is_reidentified
            else:
                # Person exists in memory (normal, no violations before)
                # Update last seen time
                self.memory_persons[best_match_id]['last_seen'] = time.time()
                
                if has_violation:
                    # NOW they have violation \u2192 Move to database!
                    if self.debug_level >= 2:
                        print(f"[ReIDManager] [!] Person M{best_match_id} NOW has violation -> Saving to DB")
                    
                    thumbnail_path = self.save_thumbnail(frame, bbox, "new", is_suspect=True)
                    
                    person_id = self.db.add_person(
                        feature_vector=features,
                        camera_id=camera_id,
                        location=self.camera_location,
                        thumbnail_path=thumbnail_path,
                        violation_status=violation_status,
                        violation_reason=violation_reason,
                        is_masked=is_masked,
                        is_helmeted=is_helmeted
                    )
                    
                    # Remove from memory, now in DB
                    del self.memory_persons[best_match_id]
                    
                    if self.debug_level >= 2:
                        print(f"[ReIDManager] [+] STATUS ESCALATION: ID={person_id} | {violation_reason}")
                    return person_id, False  # New to database, not reidentified
                else:
                    # Still no violation, keep in memory
                    if self.debug_level >= 3:
                        print(f"[ReIDManager] [+] MATCHED (Memory): Temp ID=M{best_match_id} | Normal (not saved) | Sim={best_similarity:.2f}")
                    return f"M{best_match_id}", False  # Memory ID
        
        # STEP 5: NEW PERSON (no match found)
        if has_violation:
            # NEW person WITH violation \u2192 Save to database
            thumbnail_path = self.save_thumbnail(frame, bbox, "new", is_suspect=True)
            
            person_id = self.db.add_person(
                feature_vector=features,
                camera_id=camera_id,
                location=self.camera_location,
                thumbnail_path=thumbnail_path,
                violation_status=violation_status,
                violation_reason=violation_reason,
                is_masked=is_masked,
                is_helmeted=is_helmeted
            )
            
            if self.debug_level >= 2:
                print(f"[ReIDManager] [NEW] NEW VIOLATION: ID={person_id} | {violation_reason}")
            return person_id, False
        else:
            # NEW person WITHOUT violation \u2192 Keep in memory only (don't save to DB)
            temp_id = self.next_temp_id
            self.next_temp_id += 1
            
            self.memory_persons[temp_id] = {
                'features': features,
                'first_seen': time.time(),
                'last_seen': time.time()
            }
            
            if self.debug_level >= 3:
                print(f"[ReIDManager] [+] NEW NORMAL person: Memory ID=M{temp_id} (not saved to DB)")
            return f"M{temp_id}", False  # Return memory ID
    
    def cleanup_memory_persons(self):
        """
        Clean up memory-only persons who haven't been seen recently
        
        CRITICAL: Only memory persons are cleaned up (temp IDs)
                  Database persons (violations) persist indefinitely
        """
        current_time = time.time()
        to_remove = []
        
        for temp_id, person_data in self.memory_persons.items():
            time_since_last_seen = current_time - person_data['last_seen']
            if time_since_last_seen > self.memory_cleanup_interval:
                to_remove.append(temp_id)
        
        for temp_id in to_remove:
            del self.memory_persons[temp_id]
            if self.debug_level >= 3:
                print(f"[ReIDManager] [CLEANUP] Memory cleanup: M{temp_id} removed (last seen {self.memory_cleanup_interval}s ago)")
    
    def update_tracking(self, tracks, frame, camera_id, is_masked_dict, is_helmeted_dict):
        """
        Update tracking with ReID
        
        Args:
            tracks: List of track dictionaries
            frame: Current video frame
            camera_id: Camera identifier
            is_masked_dict: Dict mapping track_id -> is_masked
            is_helmeted_dict: Dict mapping track_id -> is_helmeted
        
        Returns:
            Updated tracks with person_id and is_reidentified flag
        """
        current_time = time.time()
        current_track_ids = set()
        
        # ✅ COLLISION PREVENTION: Track which person IDs are already assigned in this frame
        # to prevent multiple tracks from getting the same ID
        assigned_person_ids = set()
        
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            current_track_ids.add(track_id)
            
            # Get mask/helmet status for this track
            is_masked = is_masked_dict.get(track_id, False)
            is_helmeted = is_helmeted_dict.get(track_id, False)
            
            # Identify person (body features, not face) - returns tuple (person_id, is_reidentified)
            # Pass assigned_person_ids to prevent collision
            result = self.identify_person(frame, bbox, camera_id, is_masked, is_helmeted, assigned_person_ids)
            
            if result is not None:
                person_id, is_reidentified = result
                track['person_id'] = person_id
                track['is_reidentified'] = is_reidentified
                
                # Mark this person_id as assigned
                if isinstance(person_id, int) or (isinstance(person_id, str) and person_id.startswith('M')):
                    assigned_person_ids.add(person_id)
                
                # Update active tracks (store bbox for spatial validation)
                self.active_tracks[track_id] = {
                    'person_id': person_id,
                    'bbox': bbox,  # Store for spatial distance validation
                    'last_seen': current_time,
                    'frames_tracked': self.active_tracks.get(track_id, {}).get('frames_tracked', 0) + 1,
                    'is_reidentified': is_reidentified
                }
            else:
                track['person_id'] = None
                track['is_reidentified'] = False
        
        # Check for exited tracks (mark scene exit for VIOLATION persons only)
        exited_tracks = []
        for track_id, track_data in list(self.active_tracks.items()):
            if track_id not in current_track_ids:
                time_since_last_seen = current_time - track_data['last_seen']
                if time_since_last_seen > self.track_exit_timeout:
                    person_id = track_data['person_id']
                    
                    # Only mark exit in DB if this is a violation person (integer ID, not memory "M#")
                    if isinstance(person_id, int):
                        self.db.mark_person_exit(person_id, camera_id, self.camera_location)
                        if self.debug_level >= 3:
                            print(f"[ReIDManager] [EXIT] VIOLATION Person {person_id} EXITED (Track {track_id} lost for {time_since_last_seen:.1f}s)")
                    else:
                        # Memory person - just forget them
                        if self.debug_level >= 3:
                            print(f"[ReIDManager] [EXIT] Memory person {person_id} left scene (Track {track_id} lost for {time_since_last_seen:.1f}s)")
                    
                    exited_tracks.append(track_id)
        
        # Clean up exited tracks
        for track_id in exited_tracks:
            del self.active_tracks[track_id]
        
        # Periodic memory cleanup (remove old memory-only persons)
        self.cleanup_memory_persons()
        
        return tracks
    
    def get_database_summary(self):
        """Get database statistics"""
        return self.db.get_summary()
    
    def close(self):
        """Close database and cleanup"""
        print(f"[ReIDManager] Closing...")
        summary = self.get_database_summary()
        print(f"[ReIDManager] Final stats: {summary}")
        self.db.close()
