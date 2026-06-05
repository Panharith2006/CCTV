import numpy as np
import cv2
import os
import time
from database.reid_database import ReIDDatabase
from detector.layer2_reid_extractor import ReIDExtractor


class ReIDManager:
    def __init__(self, db_config=None, similarity_threshold=0.7, thumbnail_dir="thumbnails", required_confirm_frames=5, required_new_frames=2, ema_alpha=0.15):
        self.db = ReIDDatabase(db_config)
        self.reid_extractor = ReIDExtractor()
        self.similarity_threshold = similarity_threshold
        self.thumbnail_dir = thumbnail_dir
        self.ema_alpha = ema_alpha
        
        # Create thumbnail directory
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # State for conservative new-person registration
        self.pending_new = {}  # track_id -> {'count': int, 'best_sim': float}
        self.required_new_frames = required_new_frames
        # State for PPE confirmation before performing ReID/enrollment
        self.pending_ppe_confirm = {}  # track_id -> {'count': int, 'attributes': dict, 'bbox': bbox, 'frame': frame}
        # Number of consecutive frames with PPE required before calling identify_person
        self.required_confirm_frames = required_confirm_frames

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
    
    def identify_person(self, frame, bbox, camera_id, track_id=None, is_masked=False, is_helmeted=False):
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
            person_id = self.db.add_person(features, camera_id, thumbnail_path=thumbnail_path, is_masked=is_masked, is_helmeted=is_helmeted)
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
            # Update using EMA alpha defined for this manager
            self.db.update_person(best_match_id, features, camera_id, is_masked=is_masked, is_helmeted=is_helmeted, alpha=self.ema_alpha)
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
            person_id = self.db.add_person(features, camera_id, thumbnail_path=thumbnail_path, is_masked=is_masked, is_helmeted=is_helmeted)
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
            # Use the attributes passed into identify_person
            person_id = self.db.add_person(features, camera_id, thumbnail_path=thumbnail_path, is_masked=is_masked, is_helmeted=is_helmeted)
            # clear pending new-person state for this track
            if track_id is not None and track_id in self.pending_new:
                del self.pending_new[track_id]
            print(f"[ReIDManager] New person registered (track {track_id}): ID={person_id} (best_sim={best_similarity:.3f})")
            return person_id

        # Not yet confident to register; return None so UI shows unknown
        return None
    
    def update_tracking(self, tracks, frame, camera_id, require_ppe=True):
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']

            attributes = track.get('attributes', {}) or {}
            has_ppe = bool(attributes.get('mask') or attributes.get('helmet'))
            # If PPE is required but not currently seen, skip ReID and clear any pending PPE confirmation
            if require_ppe and not has_ppe:
                if track_id in self.pending_ppe_confirm:
                    del self.pending_ppe_confirm[track_id]
                track['person_id'] = None
                print(f"[ReIDManager] Skipping track {track_id} because no mask/helmet was detected")
                continue

            # If we have PPE, require it to persist for several frames before performing ReID/enrollment
            if has_ppe:
                entry = self.pending_ppe_confirm.get(track_id, {'count': 0, 'attributes': attributes, 'bbox': bbox, 'frame': frame.copy()})
                # If attributes changed (e.g., mask->helmet), reset the counter
                if entry.get('attributes') != attributes:
                    entry = {'count': 0, 'attributes': attributes, 'bbox': bbox, 'frame': frame.copy()}

                entry['count'] = entry.get('count', 0) + 1
                entry['bbox'] = bbox
                entry['frame'] = frame.copy()
                self.pending_ppe_confirm[track_id] = entry

                print(f"[ReIDManager] Track {track_id} PPE confirmation: {entry['count']}/{self.required_confirm_frames}")

                if entry['count'] < self.required_confirm_frames:
                    # Not yet confirmed long enough; do not identify or enroll
                    track['person_id'] = None
                    continue

                # Confirmed PPE persistence: proceed to identify

            # Identify person (pass track_id for conservative registration)
            person_id = self.identify_person(
                frame,
                bbox,
                camera_id,
                track_id=track_id,
                is_masked=bool(attributes.get('mask')),
                is_helmeted=bool(attributes.get('helmet'))
            )

            if person_id is not None:
                track['person_id'] = person_id
                # Log detection
                self.db.add_detection(person_id, camera_id, bbox, track_id)
            else:
                track['person_id'] = None
        
        return tracks
    
    def close(self):
        self.db.close()
