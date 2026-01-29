import numpy as np
import cv2
import os
import time
from database.reid_database import ReIDDatabase
from detector.layer2_reid_extractor import ReIDExtractor


class ReIDManager:
    def __init__(self, db_config=None, similarity_threshold=0.7, thumbnail_dir="thumbnails"):
        self.db = ReIDDatabase(db_config)
        self.reid_extractor = ReIDExtractor()
        self.similarity_threshold = similarity_threshold
        self.thumbnail_dir = thumbnail_dir
        
        # Create thumbnail directory
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # State for conservative new-person registration
        self.pending_new = {}  # track_id -> {'count': int, 'best_sim': float}
        self.required_new_frames = 2

        print(f"[ReIDManager] Initialized with threshold={similarity_threshold}")
        print(f"[ReIDManager] Thumbnails will be saved to: {thumbnail_dir}")
    
    def save_thumbnail(self, frame, bbox, person_id):
        """Save person crop as thumbnail"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            filename = f"person_{person_id:03d}_{int(time.time()*1000)}.jpg"
            filepath = os.path.join(self.thumbnail_dir, filename)
            cv2.imwrite(filepath, crop)
            return filepath
        except Exception as e:
            print(f"[ReIDManager] Failed to save thumbnail: {e}")
            return None
    
    def identify_person(self, frame, bbox, camera_id, track_id=None):
        """
        Identify person from frame crop
        Returns: person_id (new or matched)
        """
        # Extract features
        features = self.reid_extractor.extract_features(frame, bbox)
        
        if features is None:
            return None
        
        # Get all known person features and compute per-person centroid
        all_features = self.db.get_all_features()
        persons = {}
        for p in all_features:
            pid = p['person_id']
            persons.setdefault(pid, []).append(p['features'])

        # If no known persons, register the first one immediately
        if len(persons) == 0:
            thumbnail_path = self.save_thumbnail(frame, bbox, 1)
            person_id = self.db.add_person(features, camera_id, thumbnail_path=thumbnail_path)
            print(f"[ReIDManager] New person registered: ID={person_id}")
            return person_id

        # Compute centroid (mean) for each person and find best match
        # Also try min-distance approach: compare to ALL stored vectors and take minimum
        best_match_id = None
        best_similarity = 0.0
        all_similarities = {}  # For detailed logging
        
        for pid, feats in persons.items():
            # Try two strategies:
            # 1. Centroid-based (average of all features)
            centroid = np.mean(feats, axis=0)
            sim_centroid = self.reid_extractor.compare_features(features, centroid)
            
            # 2. Min-distance (compare to each stored vector, take max similarity)
            # This helps when person has different appearances (masked/helmeted variants)
            sim_max = max([self.reid_extractor.compare_features(features, f) for f in feats])
            
            # Use the better of the two strategies
            similarity = max(sim_centroid, sim_max)
            
            all_similarities[pid] = similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = pid
        
        # Log all similarities for debugging/tuning
        sim_str = ", ".join([f"P{pid:03d}={sim:.3f}" for pid, sim in sorted(all_similarities.items())])
        print(f"[ReIDManager] Similarities: {sim_str} | Best: P{best_match_id:03d}={best_similarity:.3f} | Threshold: {self.similarity_threshold}")

        # If similarity is high enough, update that person
        if best_similarity >= self.similarity_threshold:
            self.db.update_person(best_match_id, features, camera_id)
            # clear pending new-person state for this track
            if track_id is not None and track_id in self.pending_new:
                del self.pending_new[track_id]
            print(f"[ReIDManager] Matched person: ID={best_match_id} (similarity={best_similarity:.3f})")
            return best_match_id

        # Not similar enough — treat conservatively: require multiple frames before registering
        if track_id is None:
            # No track context: create new person (fallback)
            next_id = max(persons.keys()) + 1
            thumbnail_path = self.save_thumbnail(frame, bbox, next_id)
            person_id = self.db.add_person(features, camera_id, thumbnail_path=thumbnail_path)
            print(f"[ReIDManager] New person registered (no track): ID={person_id} (best_sim={best_similarity:.3f})")
            return person_id

        # Update pending counter
        entry = self.pending_new.get(track_id, {'count': 0, 'best_sim': best_similarity})
        entry['count'] = entry.get('count', 0) + 1
        entry['best_sim'] = max(entry.get('best_sim', 0.0), best_similarity)
        entry['frame'] = frame.copy()  # Save frame for thumbnail
        entry['bbox'] = bbox
        self.pending_new[track_id] = entry

        print(f"[ReIDManager] Pending new for track {track_id}: count={entry['count']} best_sim={entry['best_sim']:.3f}")

        if entry['count'] >= self.required_new_frames:
            next_id = max(persons.keys()) + 1
            thumbnail_path = self.save_thumbnail(entry['frame'], entry['bbox'], next_id)
            person_id = self.db.add_person(features, camera_id, thumbnail_path=thumbnail_path)
            # clear pending new-person state for this track
            if track_id is not None and track_id in self.pending_new:
                del self.pending_new[track_id]
            print(f"[ReIDManager] New person registered (track {track_id}): ID={person_id} (best_sim={best_similarity:.3f})")
            return person_id

        # Not yet confident to register; return None so UI shows unknown
        return None
    
    def update_tracking(self, tracks, frame, camera_id):
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']

            # Identify person (pass track_id for conservative registration)
            person_id = self.identify_person(frame, bbox, camera_id, track_id=track_id)
            
            if person_id is not None:
                track['person_id'] = person_id
                
                # Log detection
                self.db.add_detection(person_id, camera_id, bbox, track_id)
            else:
                track['person_id'] = None
        
        return tracks
    
    def close(self):
        self.db.close()
