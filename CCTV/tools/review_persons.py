
# Person Database Management Tool
import cv2
import numpy as np
from database.reid_database import ReIDDatabase
from detector.layer2_reid_extractor import ReIDExtractor


def display_thumbnail(img_path, person_info):
    """Display person thumbnail with info"""
    try:
        img = cv2.imread(img_path)
        if img is not None:
            # Add text overlay
            cv2.putText(img, f"ID: {person_info[0]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Person Thumbnail", img)
            cv2.waitKey(1)
            return True
    except Exception:
        pass
    return False


def list_all_persons(db):
    """List all persons in database"""
    persons = db.get_all_persons()
    
    print("\n" + "="*80)
    print(" "*25 + "ALL PERSONS IN DATABASE")
    print("="*80)
    print(f"{'ID':<5} {'Name':<20} {'Appearances':<12} {'First Seen':<20} {'Thumbnail'}")
    print("-"*80)
    
    for p in persons:
        person_id, first_seen, last_seen, count, name, thumbnail = p
        name_str = name or "(unnamed)"
        thumb_str = "✓" if thumbnail else "✗"
        print(f"{person_id:<5} {name_str:<20} {count:<12} {first_seen:<20} {thumb_str}")
    
    print("-"*80)
    print(f"Total: {len(persons)} persons\n")
    
    return persons


def find_similar_persons(db, reid_extractor, similarity_threshold=0.75):
    """Find potentially duplicate persons"""
    print("\n[INFO] Analyzing person similarities...\n")
    
    # Get all features per person
    all_features = db.get_all_features()
    persons = {}
    for p in all_features:
        pid = p['person_id']
        persons.setdefault(pid, []).append(p['features'])
    
    # Compute centroids
    centroids = {}
    for pid, feats in persons.items():
        centroids[pid] = np.mean(feats, axis=0)
    
    # Find similar pairs
    similar_pairs = []
    person_ids = sorted(centroids.keys())
    
    for i, pid1 in enumerate(person_ids):
        for pid2 in person_ids[i+1:]:
            sim = reid_extractor.compare_features(centroids[pid1], centroids[pid2])
            if sim >= similarity_threshold:
                similar_pairs.append((pid1, pid2, sim))
    
    if similar_pairs:
        print("="*60)
        print(" "*15 + "SIMILAR PERSONS FOUND")
        print("="*60)
        print(f"{'Person 1':<12} {'Person 2':<12} {'Similarity'}")
        print("-"*60)
        for pid1, pid2, sim in similar_pairs:
            print(f"{pid1:<12} {pid2:<12} {sim:.3f}")
        print("-"*60)
        print(f"\nFound {len(similar_pairs)} potentially duplicate pairs")
        print(f"(Threshold: {similarity_threshold})\n")
    else:
        print("✓ No similar persons found\n")
    
    return similar_pairs


def merge_persons_interactive(db):
    """Interactive person merging"""
    source_id = input("\nEnter source person ID to merge: ").strip()
    target_id = input("Enter target person ID (keep this one): ").strip()
    
    try:
        source_id = int(source_id)
        target_id = int(target_id)
    except ValueError:
        print("[ERROR] Invalid person IDs")
        return
    
    # Confirm
    confirm = input(f"\n⚠️  Merge Person {source_id} into Person {target_id}? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        db.merge_persons(source_id, target_id)
        print(f"✓ Successfully merged Person {source_id} into Person {target_id}\n")
    else:
        print("[INFO] Merge cancelled\n")


def label_person_interactive(db):
    """Interactive person labeling"""
    person_id = input("\nEnter person ID to label: ").strip()
    
    try:
        person_id = int(person_id)
    except ValueError:
        print("[ERROR] Invalid person ID")
        return
    
    name = input("Enter person's name (e.g., 'Rith'): ").strip()
    
    if name:
        db.update_person_name(person_id, name)
        print(f"✓ Person {person_id} labeled as: {name}\n")
    else:
        print("[INFO] Label cancelled\n")


def delete_person_interactive(db):
    """Interactive person deletion"""
    person_id = input("\nEnter person ID to delete: ").strip()
    
    try:
        person_id = int(person_id)
    except ValueError:
        print("[ERROR] Invalid person ID")
        return
    
    confirm = input(f"\n⚠️  Delete Person {person_id}? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        db.delete_person(person_id)
        print(f"✓ Person {person_id} deleted\n")
    else:
        print("[INFO] Deletion cancelled\n")


def view_thumbnails(db):
    """View person thumbnails"""
    persons = db.get_all_persons()
    
    print("\n[INFO] Press any key to view next, ESC to exit\n")
    
    for p in persons:
        person_id, first_seen, last_seen, count, name, thumbnail = p
        
        print(f"\nPerson ID: {person_id}")
        print(f"Name: {name or '(unnamed)'}")
        print(f"Appearances: {count}")
        print(f"First seen: {first_seen}")
        
        if thumbnail and display_thumbnail(thumbnail, p):
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
        else:
            print("  (No thumbnail available)")
            input("  Press Enter to continue...")
    
    cv2.destroyAllWindows()


def main():
    """Main review interface"""
    print("\\n[INFO] Using MySQL database from config/mysql_config.py")
    
    db = ReIDDatabase()
    reid = ReIDExtractor()
    
    while True:
        print("\n" + "="*60)
        print(" "*15 + "PERSON DATABASE MANAGER")
        print("="*60)
        print("\nOptions:")
        print("  1. List all persons")
        print("  2. Find similar persons (potential duplicates)")
        print("  3. Merge persons")
        print("  4. Label person (add name)")
        print("  5. Delete person")
        print("  6. View thumbnails")
        print("  7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            list_all_persons(db)
        
        elif choice == "2":
            threshold = input("Enter similarity threshold (default: 0.75): ").strip()
            threshold = float(threshold) if threshold else 0.75
            find_similar_persons(db, reid, threshold)
        
        elif choice == "3":
            list_all_persons(db)
            merge_persons_interactive(db)
        
        elif choice == "4":
            list_all_persons(db)
            label_person_interactive(db)
        
        elif choice == "5":
            list_all_persons(db)
            delete_person_interactive(db)
        
        elif choice == "6":
            view_thumbnails(db)
        
        elif choice == "7":
            print("\n[INFO] Exiting...")
            break
        
        else:
            print("\n[ERROR] Invalid option")
    
    db.close()


if __name__ == "__main__":
    main()
