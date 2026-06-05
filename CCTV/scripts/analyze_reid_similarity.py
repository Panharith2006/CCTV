"""
Analyze ReID Feature Similarity Distribution
Helps you find the optimal similarity threshold for your system

This script:
1. Checks existing persons in database
2. Extracts their features
3. Compares all pairs to show similarity distribution
4. Recommends optimal threshold based on your data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from database.reid_database import ReIDDatabase
from detector.layer2_reid_extractor_enhanced import EnhancedReIDExtractor

def cosine_similarity(feat1, feat2):
    """Calculate cosine similarity between two feature vectors"""
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def analyze_similarity_distribution():
    """Analyze similarity scores between all pairs of persons in database"""
    
    db = ReIDDatabase()
    extractor = EnhancedReIDExtractor()
    
    print("="*80)
    print("REID FEATURE SIMILARITY ANALYSIS")
    print("="*80)
    
    # Get all persons with features
    all_data = db.get_all_features(same_day_only=False)
    
    if len(all_data) == 0:
        print("\n❌ No persons in database yet!")
        print("   Run the system and detect some violators first")
        return
    
    if len(all_data) == 1:
        print(f"\n⚠️  Only 1 person in database (ID: {all_data[0]['person_id']})")
        print("   Need at least 2 persons to analyze similarities")
        return
    
    print(f"\nFound {len(all_data)} persons in database")
    print("Calculating pairwise similarities...\n")
    
    # Calculate all pairwise similarities
    similarities = []
    pairs = []
    
    for i in range(len(all_data)):
        for j in range(i+1, len(all_data)):
            person1 = all_data[i]
            person2 = all_data[j]
            
            sim = cosine_similarity(person1['features'], person2['features'])
            similarities.append(sim)
            pairs.append((person1['person_id'], person2['person_id'], sim))
    
    similarities = np.array(similarities)
    
    # Statistics
    print("="*80)
    print("SIMILARITY STATISTICS")
    print("="*80)
    print(f"Total comparisons: {len(similarities)}")
    print(f"Mean similarity:   {similarities.mean():.4f}")
    print(f"Std deviation:     {similarities.std():.4f}")
    print(f"Min similarity:    {similarities.min():.4f}")
    print(f"Max similarity:    {similarities.max():.4f}")
    print(f"Median:            {np.median(similarities):.4f}")
    
    # Show distribution
    print("\n" + "="*80)
    print("SIMILARITY DISTRIBUTION")
    print("="*80)
    
    ranges = [
        (0.0, 0.3, "Very Different"),
        (0.3, 0.4, "Different"),
        (0.4, 0.5, "Somewhat Similar"),
        (0.5, 0.6, "Similar"),
        (0.6, 0.7, "Very Similar"),
        (0.7, 0.8, "Highly Similar"),
        (0.8, 1.0, "Almost Identical"),
    ]
    
    for low, high, label in ranges:
        count = np.sum((similarities >= low) & (similarities < high))
        percentage = (count / len(similarities)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{low:.1f}-{high:.1f} {label:20s} | {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Show top matches (potential same person)
    print("\n" + "="*80)
    print("TOP 10 HIGHEST SIMILARITIES (Potential Same Person)")
    print("="*80)
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    
    for i, (p1, p2, sim) in enumerate(pairs_sorted[:10], 1):
        match_status = "✓ MATCH" if sim >= 0.62 else "✗ No match"
        print(f"{i:2d}. P{p1:03d} vs P{p2:03d}: {sim:.4f} {match_status}")
    
    # Show bottom matches (definitely different)
    print("\n" + "="*80)
    print("BOTTOM 10 LOWEST SIMILARITIES (Definitely Different)")
    print("="*80)
    
    for i, (p1, p2, sim) in enumerate(pairs_sorted[-10:], 1):
        print(f"{i:2d}. P{p1:03d} vs P{p2:03d}: {sim:.4f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*80)
    
    print("\nCurrent threshold: 0.62")
    print("\nBased on your data:")
    
    # Conservative (high precision)
    conservative = np.percentile(similarities, 75)
    print(f"\n1. CONSERVATIVE (Fewer false matches, may miss re-IDs)")
    print(f"   Threshold: {conservative:.2f}")
    print(f"   Would match: {np.sum(similarities >= conservative)} pairs")
    
    # Balanced
    balanced = np.percentile(similarities, 50)
    print(f"\n2. BALANCED (Good trade-off)")
    print(f"   Threshold: {balanced:.2f}")
    print(f"   Would match: {np.sum(similarities >= balanced)} pairs")
    
    # Aggressive (high recall)
    aggressive = np.percentile(similarities, 25)
    print(f"\n3. AGGRESSIVE (More re-IDs, may have false matches)")
    print(f"   Threshold: {aggressive:.2f}")
    print(f"   Would match: {np.sum(similarities >= aggressive)} pairs")
    
    # Current threshold analysis
    current_threshold = 0.62
    current_matches = np.sum(similarities >= current_threshold)
    print(f"\n4. CURRENT THRESHOLD: {current_threshold:.2f}")
    print(f"   Would match: {current_matches} pairs ({(current_matches/len(similarities)*100):.1f}%)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR VIOLATION-ONLY SYSTEM")
    print("="*80)
    print("\nFor tracking violators, prioritize accuracy over recall:")
    print("  ✓ Use CONSERVATIVE threshold (0.62-0.70)")
    print("  ✓ Better to have multiple IDs for same person")
    print("  ✓ Than to wrongly match different people")
    
    if similarities.max() > 0.8:
        print("\n⚠️  WARNING: Some pairs have very high similarity (>0.8)")
        print("   These might be the same person appearing multiple times")
        print("   Consider using threshold 0.60-0.65 to catch these re-IDs")
    
    if similarities.mean() < 0.4:
        print("\n✓ Good: Low average similarity means distinct persons")
        print("  Current threshold 0.62 is appropriate")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        analyze_similarity_distribution()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
