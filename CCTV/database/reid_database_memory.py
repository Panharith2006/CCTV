"""
Simple in-memory database for ReID system
Use this when MySQL is not available

Storage Strategy:
- Save ONLY when: first detected, flagged as suspect, or exits scene
- Keep ONE ID per person (even if mask status changes)
- Track movement across cameras
- Store images for suspect persons only
"""
import numpy as np
from datetime import datetime
import json


class ReIDDatabase:
    def __init__(self, config=None):
        """Initialize in-memory database"""
        self.persons = {}  # person_id -> person_data
        self.location_history = {}  # person_id -> list of location records
        self.suspect_images = {}  # person_id -> list of image paths
        self.next_person_id = 1
        print("[Database] Using in-memory database (no MySQL required)")
        print("[Database] Storage strategy: Save only on first detection, suspect flag, or scene exit")
    
    def add_person(self, feature_vector, camera_id, location, thumbnail_path=None, name=None, is_masked=False, is_helmeted=False):
        """
        Add new person to database (ONLY on first detection)
        
        Args:
            feature_vector: 128D feature vector from ReID extractor
            camera_id: Camera identifier
            location: Camera location (e.g., "Entrance", "Hallway")
            thumbnail_path: Path to saved thumbnail image
            name: Optional person name
            is_masked: Whether person is wearing mask
            is_helmeted: Whether person is wearing helmet
        
        Returns:
            person_id: New unique person ID
        """
        person_id = self.next_person_id
        self.next_person_id += 1
        
        now = datetime.now()
        
    def update_person_location(self, person_id, camera_id, location, is_masked=False, is_helmeted=False):
        """
        Update person's location when they move to a new camera
        (Does NOT save continuously - only tracks movement)
        
        Args:
            person_id: Person identifier
            camera_id: New camera identifier
            location: New location
            is_masked: Current mask status
            is_helmeted: Current helmet status
        """
        if person_id not in self.persons:
            return
        
        now = datetime.now()
        
        # Update person record
        self.persons[person_id]['last_seen_time'] = now
        self.persons[person_id]['last_camera_id'] = camera_id
        self.persons[person_id]['last_location'] = location
        self.persons[person_id]['is_masked'] = is_masked
        self.persons[person_id]['is_helmeted'] = is_helmeted
        
        # Add to location history (track movement across cameras)
        if person_id not in self.location_history:
            self.location_history[person_id] = []
        
        # Only add if camera changed
        if not self.location_history[person_id] or self.location_history[person_id][-1]['camera_id'] != camera_id:
            self.location_history[person_id].append({
                'camera_id': camera_id,
                'location': location,
                'timestamp': now,
                'is_masked': is_masked,
                'is_helmeted': is_helmeted
            })
            print(f"[Database] Person {person_id} moved to {location} (Camera: {camera_id})")
    
    def mark_person_as_suspect(self, person_id, reason="abnormal_behavior"):
        """
        Mark person as suspect (triggers image storage)
        
        Args:
            person_id: Person identifier
            reason: Reason for suspect flag
        """
        if person_id not in self.persons:
            return
        
        self.persons[person_id]['status'] = 'suspect'
        self.persons[person_id]['suspect_reason'] = reason
        self.persons[person_id]['number_of_sequences_saved'] += 1
        
        print(f"[Database] Person {person_id} marked as SUSPECT: {reason}")
    
    def save_suspect_image(self, person_id, image_path, timestamp=None):
        """
        Save image for suspect person only
        
        Args:
            person_id: Person identifier
            image_path: Path to saved image
            timestamp: When image was captured
        """
        if person_id not in self.suspect_images:
            self.suspect_images[person_id] = []
        
        self.suspect_images[person_id].append({
            'image_path': image_path,
            'timestamp': timestamp or datetime.now()
        })
        
        print(f"[Database] Suspect image saved for Person {person_id}: {image_path}")
    
    def mark_person_exit(self, person_id, camera_id, location):
        """
        Mark person as exited (final storage point)
        
        Args:
            person_id: Person identifier
            camera_id: Camera where exit detected
            location: Location name
        """
        if person_id not in self.persons:
            return
        
        self.persons[person_id]['status'] = 'exited'
        self.persons[person_id]['last_seen'] = datetime.now()
        
        # Add final exit location
        if person_id not in self.location_history:
            self.location_history[person_id] = []
        
        self.location_history[person_id].append({
            'camera_id': camera_id,
            'location': location,
            'timestamp': datetime.now(),
            'event': 'exit'
        })
        
        print(f"[Database] Person {person_id} EXITED scene at {location}")
    
    def get_all_features(self):
        """
        Get all person IDs and their 128D feature vectors
        (Only for active/normal persons - not exited)
        
        Returns:
            List of dicts with 'person_id' and 'features' (128D vector)
        """
        results = []
        for person_id, person_data in self.persons.items():
            # Only return features for persons still in scene
            if person_data['status'] not in ['exited']:
                results.append({
                    'person_id': person_id,
                    'features': np.array(person_data['128d_feature_vector'])
                })
        return results
    
    def get_person_location_history(self, person_id):
        """Get movement history across cameras"""
        return self.location_history.get(person_id, [])
    
    def get_person_info(self, person_id):
        """Get all features for a person"""
        features = [f['feature_vector'] for f in self.features.values() 
                   if f['person_id'] == person_id]
        return features
    
    def get_all_features(self):
        """Get all person IDs and their features - returns list of dicts"""
        results = []
        for f in self.features.values():
            # Only return features for active persons
            if f['person_id'] in self.persons and self.persons[f['person_id']]['status'] == 'active':
                results.append({
                    'person_id': f['person_id'],
                    'features': f['feature_vector']
                })
        return results
    
    def get_person_location_history(self, person_id):
        """Get movement history across cameras"""
        return self.location_history.get(person_id, [])
    
    def get_person_info(self, person_id):
        """Get complete person information"""
        if person_id not in self.persons:
            return None
        
        person_data = self.persons[person_id].copy()
        person_data['location_history'] = self.get_person_location_history(person_id)
        
        if person_id in self.suspect_images:
            person_data['suspect_images'] = self.suspect_images[person_id]
        
        return person_data
    
    def get_person_stats(self, person_id):
        """Get person statistics (for backward compatibility)"""
        if person_id not in self.persons:
            return None
        p = self.persons[person_id]
        return (p['person_id'], p['first_seen_time'], p['last_seen_time'], 
                p.get('name'), p['number_of_sequences_saved'], p.get('thumbnail_path'))
    
    def get_person_name(self, person_id):
        """Get person name"""
        if person_id in self.persons:
            return self.persons[person_id].get('name')
        return None
    
    def set_person_name(self, person_id, name):
        """Set person name"""
        if person_id in self.persons:
            self.persons[person_id]['name'] = name
            print(f"[Database] Person {person_id} named: {name}")
    
    def get_all_persons(self):
        """Get all active persons (not exited)"""
        return [(p['person_id'], p.get('name'), p['status']) 
                for p in self.persons.values() 
                if p['status'] != 'exited']
    
    def get_summary(self):
        """Get database summary statistics"""
        total = len(self.persons)
        active = len([p for p in self.persons.values() if p['status'] in ['normal', 'suspect']])
        suspects = len([p for p in self.persons.values() if p['status'] == 'suspect'])
        exited = len([p for p in self.persons.values() if p['status'] == 'exited'])
        
        return {
            'total_persons': total,
            'active_persons': active,
            'suspects': suspects,
            'exited': exited,
            'total_suspect_images': sum(len(imgs) for imgs in self.suspect_images.values())
        }
    
    def close(self):
        """Close database connection"""
        summary = self.get_summary()
        print(f"[Database] Closing. Summary: {summary}")
