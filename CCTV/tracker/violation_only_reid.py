"""
Violation-Only Re-Identification Manager

SIMPLIFIED STRATEGY:
- Normal persons (no mask/helmet): Tracked temporarily with track_id (for motion/behavior analysis),
  NO person_id assigned, NOT saved to database, forgotten when they leave
- Violators (mask/helmet): Assigned person_id, saved to database, re-identified on return

Workflow:
1. All persons tracked with track_id (needed for motion analysis, temporal smoothing)
2. Mask/Helmet detection on head regions
3. If violation detected:
   → Extract features → Match against existing violators in database
   → If match found → Reuse existing person_id
   → If no match → Assign new person_id and save to database
4. If NO violation detected:
   → Track continues with track_id only (for loitering/motion detection)
   → No person_id assigned
   → Not saved to database
   → Completely forgotten when person exits

Key Point: track_id is temporary (for active tracking), person_id is permanent (for violators only)
"""

import numpy as np
import cv2
import os
import time
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime

from detector.layer2_reid_extractor_enhanced import EnhancedReIDExtractor
from database.reid_database import ReIDDatabase


class ViolationOnlyReIDManager:
    """Minimal ReID manager that ONLY tracks persons with violations (mask/helmet)"""
    
    def __init__(self,
                 db_config=None,
                 similarity_threshold=0.62,  # Increased for more confident matching (was 0.55)
                 thumbnail_dir="thumbnails",
                 camera_id="camera_1",
                 camera_location=None,
                 debug_level=1):
        """
        Initialize violation-only ReID manager.
        
        Args:
            db_config: Database configuration
            similarity_threshold: Threshold for matching violators
            thumbnail_dir: Directory for saving thumbnails
            camera_id: Camera identifier
            camera_location: Camera location name
            debug_level: Debug output level
        """
        self.camera_id = camera_id
        self.camera_location = camera_location or camera_id
        self.similarity_threshold = similarity_threshold
        self.thumbnail_dir = thumbnail_dir
        self.debug_level = debug_level
        
        # ReID feature extractor
        self.reid_extractor = EnhancedReIDExtractor(quality_threshold=0.25)
        
        # Database for violation persons only
        self.use_database = db_config is not None
        if self.use_database:
            try:
                self.db = ReIDDatabase(db_config)
                print(f"[ViolationReID] Database connected: VIOLATION-ONLY storage")
            except Exception as e:
                print(f"[ViolationReID] Database connection failed: {e}")
                self.db = None
                self.use_database = False
        else:
            self.db = None
        
        # Active tracking
        self.active_tracks = {}  # track_id -> {person_id, features, bbox, last_seen, has_violation}
        self.track_to_person = {}  # track_id -> person_id (only for violators)
        
        # Statistics
        self.frame_count = 0
        self.total_violators = 0
        self.total_normal = 0
        
        os.makedirs(thumbnail_dir, exist_ok=True)
        os.makedirs(os.path.join(thumbnail_dir, "suspects"), exist_ok=True)
        
        print(f"[ViolationReID] Initialized")
        print(f"[ViolationReID] Strategy: VIOLATION-ONLY (mask/helmet → database, normal → ignored)")
        print(f"[ViolationReID] Similarity threshold: {similarity_threshold}")
        print(f"[ViolationReID] Camera: {camera_id} @ {self.camera_location}")
    
    def identify_person(self,
                       frame: np.ndarray,
                       bbox: List[float],
                       track_id: int,
                       is_masked: bool = False,
                       is_helmeted: bool = False) -> Tuple[Optional[int], bool, float]:
        """
        Identify person ONLY if they have a violation.
        
        Note: All persons are tracked with track_id (for motion/behavior analysis),
        but only violators get persistent person_id (for database storage & re-identification).
        
        Args:
            frame: Current frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID (temporary, used during active tracking)
            is_masked: Has mask detected
            is_helmeted: Has helmet detected
            
        Returns:
            (person_id, is_reidentified, confidence)
            - person_id: None for normal persons, int for violators (persistent identity)
            - is_reidentified: True if recognized violator
            - confidence: Matching confidence
        """
        has_violation = is_masked or is_helmeted
        
        # CASE 1: No violation → Track continues with track_id only
        # No person_id assigned, no database save, forgotten when they exit
        if not has_violation:
            self.total_normal += 1
            return None, False, 0.0
        
        # CASE 2: Has violation → Need to identify
        self.total_violators += 1
        
        # Log violation
        if self.debug_level >= 2:
            violation_type = []
            if is_helmeted:
                violation_type.append("HELMET")
            if is_masked:
                violation_type.append("MASK")
            print(f"[ViolationReID] Track {track_id}: VIOLATION detected - {'+'.join(violation_type)}")
        
        # Extract features
        features, quality = self.reid_extractor.extract_features(frame, bbox, return_quality=True)
        
        if features is None:
            # Force extraction for violations (lower quality threshold)
            if self.debug_level >= 2:
                print(f"[ViolationReID] Track {track_id}: Low quality, forcing extraction...")
            original_threshold = self.reid_extractor.quality_threshold
            self.reid_extractor.quality_threshold = 0.0
            features, quality = self.reid_extractor.extract_features(frame, bbox, return_quality=True)
            self.reid_extractor.quality_threshold = original_threshold
            
            if features is None:
                if self.debug_level >= 1:
                    print(f"[ViolationReID] Track {track_id}: Feature extraction failed!")
                return None, False, 0.0
        
        # Try to match with existing violators in database
        person_id = None
        confidence = 0.0
        is_reidentified = False
        
        if self.use_database and self.db:
            # Load violators from database (same-day only)
            existing_violators = self.db.get_all_features(same_day_only=True)
            
            if len(existing_violators) > 0:
                # Compare with existing violators
                best_match_id = None
                best_similarity = 0.0
                
                for violator in existing_violators:
                    vid = violator['person_id']
                    vfeatures = violator['features']
                    
                    # Skip if already assigned to another track
                    if vid in self.track_to_person.values() and self.track_to_person.get(track_id) != vid:
                        continue
                    
                    similarity = np.dot(features, vfeatures)
                    similarity = np.clip(similarity, 0.0, 1.0)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = vid
                
                # Check if match is good enough
                if best_similarity >= self.similarity_threshold:
                    person_id = best_match_id
                    confidence = best_similarity
                    is_reidentified = True
                    
                    if self.debug_level >= 2:
                        print(f"[ViolationReID] Track {track_id}: MATCHED existing violator P{person_id:03d} (conf={confidence:.3f})")
                    
                    # Update database
                    try:
                        self.db.update_person(
                            person_id=person_id,
                            feature_vector=features,
                            camera_id=self.camera_id,
                            is_masked=is_masked,
                            is_helmeted=is_helmeted,
                            mark_reidentified=True
                        )
                    except Exception as e:
                        if self.debug_level >= 1:
                            print(f"[ViolationReID] DB update failed: {e}")
        
        # Create new violator if no match found
        if person_id is None:
            # Save thumbnail
            try:
                x1, y1, x2, y2 = map(int, bbox)
                person_img = frame[y1:y2, x1:x2]
                timestamp = int(time.time() * 1000)
                thumbnail_filename = f"suspect_{timestamp}_{track_id}.jpg"
                thumbnail_path = os.path.join(self.thumbnail_dir, "suspects", thumbnail_filename)
                cv2.imwrite(thumbnail_path, person_img)
            except Exception:
                thumbnail_path = None
            
            # Determine violation status
            if is_helmeted:
                violation_status = "ALERT"
                violation_reason = "HELMET"
                if is_masked:
                    violation_reason = "HELMET+MASK"
            elif is_masked:
                violation_status = "WARNING"
                violation_reason = "MASK"
            else:
                violation_status = "WARNING"
                violation_reason = "UNKNOWN"
            
            # Save to database
            if self.use_database and self.db:
                try:
                    person_id = self.db.add_person(
                        feature_vector=features,
                        camera_id=self.camera_id,
                        location=self.camera_location,
                        thumbnail_path=thumbnail_path,
                        violation_status=violation_status,
                        violation_reason=violation_reason,
                        is_masked=is_masked,
                        is_helmeted=is_helmeted
                    )
                    confidence = 1.0
                    
                    if self.debug_level >= 2:
                        print(f"[ViolationReID] Track {track_id}: NEW violator P{person_id:03d} created ({violation_reason})")
                
                except Exception as e:
                    if self.debug_level >= 1:
                        print(f"[ViolationReID] DB insert failed: {e}")
                    person_id = None
        
        return person_id, is_reidentified, confidence
    
    def update_tracks(self,
                     frame: np.ndarray,
                     tracks: List[Dict],
                     is_masked_dict: Dict[int, bool],
                     is_helmeted_dict: Dict[int, bool]) -> List[Dict]:
        """
        Update tracks with person IDs (only for violators).
        
        Args:
            frame: Current frame
            tracks: List of track dictionaries
            is_masked_dict: Dict mapping track_id -> is_masked
            is_helmeted_dict: Dict mapping track_id -> is_helmeted
            
        Returns:
            Updated tracks with person_id (only for violators)
        """
        self.frame_count += 1
        current_track_ids = set()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            current_track_ids.add(track_id)
            
            is_masked = is_masked_dict.get(track_id, False)
            is_helmeted = is_helmeted_dict.get(track_id, False)
            
            # Identify (only violators get person_id)
            person_id, is_reid, confidence = self.identify_person(
                frame, bbox, track_id, is_masked, is_helmeted
            )
            
            # Update track
            track['person_id'] = person_id  # None for normal persons, int for violators
            track['is_reidentified'] = is_reid
            track['confidence'] = confidence
            
            # Update active tracking
            self.active_tracks[track_id] = {
                'person_id': person_id,
                'bbox': bbox,
                'has_violation': is_masked or is_helmeted,
                'last_seen': time.time()
            }
            
            if person_id is not None:
                self.track_to_person[track_id] = person_id
        
        # Handle lost tracks
        lost_tracks = set(self.active_tracks.keys()) - current_track_ids
        for track_id in lost_tracks:
            track_info = self.active_tracks[track_id]
            person_id = track_info.get('person_id')
            
            # Mark exit for violators only (save to database)
            if person_id is not None and self.use_database and self.db:
                try:
                    self.db.mark_person_exit(person_id, self.camera_id, self.camera_location)
                    if self.debug_level >= 2:
                        print(f"[ViolationReID] Violator P{person_id:03d} exited (Track {track_id})")
                except Exception:
                    pass
            # Normal persons (person_id=None): No exit logging, just forgotten
            
            # Clean up tracking state (both normal and violators)
            del self.active_tracks[track_id]
            if track_id in self.track_to_person:
                del self.track_to_person[track_id]
        
        return tracks
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'frames_processed': self.frame_count,
            'active_tracks': len(self.active_tracks),
            'total_violators': self.total_violators,
            'total_normal': self.total_normal,
            'database_enabled': self.use_database
        }
    
    def close(self):
        """Close database connection"""
        if self.db:
            try:
                self.db.close()
            except Exception:
                pass
