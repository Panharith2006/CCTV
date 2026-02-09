"""
Visual Test: ReID Database Comparison Logic

This script helps you understand how the system compares 
new detections with existing persons in database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.reid_database import ReIDDatabase
from config.mysql_config import MYSQL_CONFIG
import numpy as np


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity (same as ReID uses)"""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-12))


def visualize_comparison(new_vector, existing_persons, threshold=0.7):
    """
    Show step-by-step how system compares new detection with database
    """
    print("\n" + "="*80)
    print("  🔍 ReID Comparison Process")
    print("="*80)
    
    print(f"\n📊 Incoming Detection:")
    print(f"   • Feature Vector: 128D array")
    print(f"   • Need to compare with: {len(existing_persons)} existing person(s)")
    print(f"   • Threshold: {threshold}")
    
    print(f"\n🔄 Comparing with each person in database...\n")
    
    best_match_id = None
    best_similarity = 0.0
    
    for i, person_data in enumerate(existing_persons, 1):
        pid = person_data['person_id']
        stored_vector = person_data['features']
        
        # Calculate similarity
        similarity = cosine_similarity(new_vector, stored_vector)
        
        # Visual representation
        bar_length = int(similarity * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        status = "✓ MATCH" if similarity >= threshold else "✗ No match"
        
        print(f"   [{i}/{len(existing_persons)}] Person ID={pid:03d}")
        print(f"       Similarity: {similarity:.4f} │{bar}│ {status}")
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = pid
    
    print("\n" + "-"*80)
    print(f"\n🎯 Best Match Result:")
    
    if best_similarity >= threshold:
        print(f"   ✅ MATCHED to existing Person ID={best_match_id}")
        print(f"   └─ Similarity: {best_similarity:.4f} >= {threshold} (threshold)")
        print(f"   └─ Action: Keep existing ID={best_match_id}")
        print(f"   └─ Database: No new entry (just update last_seen)")
        return best_match_id, "MATCHED"
    else:
        new_id = max([p['person_id'] for p in existing_persons]) + 1 if existing_persons else 1
        print(f"   ❌ NO MATCH found")
        print(f"   └─ Best similarity: {best_similarity:.4f} < {threshold} (threshold)")
        print(f"   └─ Action: Create NEW Person ID={new_id}")
        print(f"   └─ Database: Save new person + feature vector")
        return new_id, "NEW"


def test_scenarios():
    """Run test scenarios showing comparison logic"""
    
    print("\n" + "="*80)
    print("  🧪 ReID Comparison Test Scenarios")
    print("="*80)
    
    try:
        db = ReIDDatabase(MYSQL_CONFIG)
        print("\n✓ Connected to MySQL database")
    except Exception as e:
        print(f"\n✗ Database error: {e}")
        print("Using simulated data instead...\n")
        db = None
    
    # Get existing persons
    if db:
        existing = db.get_all_features()
        db.close()
    else:
        # Simulate existing persons
        existing = [
            {'person_id': 1, 'features': np.random.rand(128)},
            {'person_id': 2, 'features': np.random.rand(128)},
            {'person_id': 3, 'features': np.random.rand(128)},
        ]
    
    if len(existing) == 0:
        print("\n⚠️  Database is empty! No persons to compare with.")
        print("\nFirst detection will be saved as ID=1 without comparison.\n")
        return
    
    print(f"\n📊 Current Database State:")
    print(f"   • Total Persons: {len(existing)}")
    for p in existing:
        print(f"   • Person ID={p['person_id']:03d} (128D vector stored)")
    
    # Scenario 1: Strong match (same person)
    print("\n" + "="*80)
    print("📍 Scenario 1: Same Person Returns (High Similarity)")
    print("="*80)
    print("Simulating: Person 1 detected again (same clothes, similar pose)")
    
    # Create similar vector (high similarity ~0.85)
    similar_vector = existing[0]['features'] + np.random.randn(128) * 0.1
    similar_vector = similar_vector / np.linalg.norm(similar_vector)
    
    result_id, result_type = visualize_comparison(similar_vector, existing)
    
    # Scenario 2: Weak match (different person)
    print("\n" + "="*80)
    print("📍 Scenario 2: New Person Detected (Low Similarity)")
    print("="*80)
    print("Simulating: Different person with different appearance")
    
    # Create different vector (low similarity ~0.45)
    different_vector = np.random.rand(128)
    different_vector = different_vector / np.linalg.norm(different_vector)
    
    result_id, result_type = visualize_comparison(different_vector, existing)
    
    # Scenario 3: Borderline case
    print("\n" + "="*80)
    print("📍 Scenario 3: Borderline Case (Near Threshold)")
    print("="*80)
    print("Simulating: Similar appearance but different person (e.g., similar clothes)")
    
    # Create borderline vector (similarity ~0.68)
    borderline_vector = existing[1]['features'] * 0.6 + np.random.rand(128) * 0.4
    borderline_vector = borderline_vector / np.linalg.norm(borderline_vector)
    
    result_id, result_type = visualize_comparison(borderline_vector, existing)
    
    print("\n" + "="*80)
    print("  📚 Key Takeaways")
    print("="*80)
    print("""
1. ✅ Every new detection compares with ALL persons in database
   └─ O(n) complexity: 10 persons = 10 comparisons
   
2. ✅ System finds BEST match among all persons
   └─ Chooses highest similarity score
   
3. ✅ Threshold determines match/no-match
   └─ >= 0.7 = Same person (keep ID)
   └─ <  0.7 = Different person (new ID)
   
4. ✅ Performance is fast even with many persons
   └─ 128D vector comparison: ~0.1-1ms per person
   └─ 100 persons = ~10-100ms total comparison time
   
5. ✅ Database grows only when NEW persons detected
   └─ Not every frame, only unique persons
""")


def show_live_database():
    """Show current database state"""
    print("\n" + "="*80)
    print("  📊 Current Database State")
    print("="*80)
    
    try:
        db = ReIDDatabase(MYSQL_CONFIG)
        existing = db.get_all_features()
        
        print(f"\n✓ Connected to database")
        print(f"✓ Total Persons: {len(existing)}")
        
        if len(existing) > 0:
            print(f"\nPersons in database:")
            for p in existing:
                print(f"   • Person ID={p['person_id']:03d}")
                print(f"     └─ Feature vector: 128D array")
                print(f"     └─ Ready for comparison")
        else:
            print("\n⚠️  Database is empty")
            print("When first person is detected, they will be saved as ID=1")
            print("All future detections will compare against this person.")
        
        db.close()
        
    except Exception as e:
        print(f"\n✗ Database error: {e}")
        print("\nMake sure:")
        print("  1. MySQL is running")
        print("  2. config/mysql_config.py is configured")
        print("  3. Run: python scripts/test_mysql_connection.py")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  🎯 ReID Database Comparison Demo")
    print("="*80)
    print("\nThis script demonstrates how the system compares new detections")
    print("with existing persons in the database.\n")
    
    # Show current database
    show_live_database()
    
    print("\n" + "="*80)
    print("Choose an option:")
    print("="*80)
    print("  1. Run test scenarios (simulated data)")
    print("  2. Show database state only")
    print("  3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            test_scenarios()
        elif choice == "2":
            show_live_database()
        elif choice == "3":
            print("\nExiting...")
        else:
            print("\nInvalid choice")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    print()
