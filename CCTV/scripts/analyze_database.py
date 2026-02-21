"""
Database Schema Checker and Optimizer
Analyzes current database structure and identifies issues
"""
import mysql.connector
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.mysql_config import MYSQL_CONFIG
    config = MYSQL_CONFIG
except ImportError:
    print("❌ MySQL config not found. Cannot check database.")
    sys.exit(1)

print("="*80)
print("DATABASE SCHEMA ANALYZER")
print("="*80)

try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    
    print(f"\n✅ Connected to database: {config['database']}")
    print("\n" + "="*80)
    print("TABLE ANALYSIS")
    print("="*80)
    
    # Check what tables exist
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"\n📊 Found {len(tables)} tables:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table:<25} ({count:>6} rows)")
    
    # Analyze persons table
    print("\n" + "="*80)
    print("PERSONS TABLE ANALYSIS")
    print("="*80)
    
    cursor.execute("DESCRIBE persons")
    columns = cursor.fetchall()
    
    print("\n📋 Current columns:")
    for col in columns:
        col_name = col[0]
        col_type = col[1]
        nullable = "NULL" if col[2] == "YES" else "NOT NULL"
        default = f"DEFAULT {col[4]}" if col[4] else ""
        print(f"  {col_name:<25} {col_type:<30} {nullable:<10} {default}")
    
    # Check for unused columns
    print("\n🔍 Checking column usage...")
    
    # Check status values
    cursor.execute("SELECT status, COUNT(*) FROM persons GROUP BY status")
    status_counts = cursor.fetchall()
    print("\n  Status distribution:")
    for status, count in status_counts:
        print(f"    {status:<15} : {count:>5} persons")
    
    # Check violation_status
    cursor.execute("SELECT violation_status, COUNT(*) FROM persons GROUP BY violation_status")
    violation_counts = cursor.fetchall()
    print("\n  Violation status distribution:")
    for vstatus, count in violation_counts:
        print(f"    {vstatus or 'NULL':<15} : {count:>5} persons")
    
    # Check if name is used
    cursor.execute("SELECT COUNT(*) FROM persons WHERE name IS NOT NULL")
    named_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM persons")
    total_count = cursor.fetchone()[0]
    print(f"\n  Named persons: {named_count}/{total_count} ({named_count/total_count*100 if total_count > 0 else 0:.1f}%)")
    
    # Check is_reidentified
    cursor.execute("SELECT COUNT(*) FROM persons WHERE is_reidentified = 1")
    reid_count = cursor.fetchone()[0]
    print(f"  Re-identified: {reid_count}/{total_count} ({reid_count/total_count*100 if total_count > 0 else 0:.1f}%)")
    
    # Analyze features table
    print("\n" + "="*80)
    print("FEATURES TABLE ANALYSIS")
    print("="*80)
    
    cursor.execute("SELECT COUNT(*) FROM features")
    feature_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT person_id) FROM features")
    unique_persons = cursor.fetchone()[0]
    
    print(f"\n  Total features: {feature_count}")
    print(f"  Unique persons: {unique_persons}")
    if unique_persons > 0:
        print(f"  Avg features/person: {feature_count/unique_persons:.1f}")
    
    # Check if is_masked/is_helmeted are used
    cursor.execute("SELECT COUNT(*) FROM features WHERE is_masked = 1")
    masked_features = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM features WHERE is_helmeted = 1")
    helmeted_features = cursor.fetchone()[0]
    
    print(f"\n  Features with mask: {masked_features}/{feature_count} ({masked_features/feature_count*100 if feature_count > 0 else 0:.1f}%)")
    print(f"  Features with helmet: {helmeted_features}/{feature_count} ({helmeted_features/feature_count*100 if feature_count > 0 else 0:.1f}%)")
    
    # Analyze detections table
    print("\n" + "="*80)
    print("DETECTIONS TABLE ANALYSIS")
    print("="*80)
    
    cursor.execute("SELECT COUNT(*) FROM detections")
    detection_count = cursor.fetchone()[0]
    print(f"\n  Total detections: {detection_count}")
    
    if detection_count > 0:
        cursor.execute("SELECT COUNT(DISTINCT person_id) FROM detections")
        unique_detected = cursor.fetchone()[0]
        print(f"  Unique persons: {unique_detected}")
        print(f"  Avg detections/person: {detection_count/unique_detected:.1f}")
        
        # Check oldest and newest
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM detections")
        oldest, newest = cursor.fetchone()
        print(f"  Date range: {oldest} to {newest}")
    
    # Analyze location_history table
    print("\n" + "="*80)
    print("LOCATION_HISTORY TABLE ANALYSIS")
    print("="*80)
    
    cursor.execute("SELECT COUNT(*) FROM location_history")
    location_count = cursor.fetchone()[0]
    print(f"\n  Total location records: {location_count}")
    
    if location_count > 0:
        cursor.execute("SELECT event_type, COUNT(*) FROM location_history GROUP BY event_type")
        event_counts = cursor.fetchall()
        print("\n  Event type distribution:")
        for event, count in event_counts:
            print(f"    {event:<20} : {count:>5} events")
    
    # Analyze suspect_images table
    print("\n" + "="*80)
    print("SUSPECT_IMAGES TABLE ANALYSIS")
    print("="*80)
    
    cursor.execute("SELECT COUNT(*) FROM suspect_images")
    image_count = cursor.fetchone()[0]
    print(f"\n  Total suspect images: {image_count}")
    
    # Compare with thumbnail_path in persons table
    cursor.execute("SELECT COUNT(*) FROM persons WHERE thumbnail_path IS NOT NULL")
    thumbnail_count = cursor.fetchone()[0]
    print(f"  Persons with thumbnail_path: {thumbnail_count}")
    
    # RECOMMENDATIONS
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check for redundant tables
    if image_count == 0 and thumbnail_count > 0:
        recommendations.append("⚠️  suspect_images table is empty but persons.thumbnail_path is used")
        recommendations.append("   → Consider removing suspect_images table (redundant)")
    
    # Check detections table
    if detection_count > feature_count * 10:
        recommendations.append("⚠️  detections table has many more rows than features")
        recommendations.append(f"   → {detection_count} detections vs {feature_count} features")
        recommendations.append("   → Consider if per-frame detection logging is necessary")
    
    # Check for unused status values
    status_values = [s[0] for s in status_counts]
    if 'merged' not in status_values and 'deleted' not in status_values:
        recommendations.append("⚠️  'merged' and 'deleted' status values are never used")
        recommendations.append("   → Consider simplifying status to just 'active' and 'exited'")
    
    # Check name usage
    if named_count == 0:
        recommendations.append("⚠️  'name' column is never used")
        recommendations.append("   → Consider removing if person naming is not a feature")
    
    if len(recommendations) == 0:
        print("\n✅ No major issues found! Database schema looks good.")
    else:
        print()
        for rec in recommendations:
            print(rec)
    
    # SUGGESTED OPTIMIZATIONS
    print("\n" + "="*80)
    print("SUGGESTED SCHEMA OPTIMIZATIONS")
    print("="*80)
    
    print("""
1. SIMPLIFY persons TABLE:
   - Remove 'name' column (unused for violation-only storage)
   - Simplify 'status' to only 'active' and 'exited'
   - Remove status values 'merged' and 'deleted' (unused)

2. OPTIMIZE detections TABLE:
   - Consider removing if not used for audit trails
   - Or add retention policy (delete old detections after 30 days)
   - Currently logs every frame which can create millions of rows

3. REMOVE suspect_images TABLE:
   - Redundant with thumbnail_path in persons table
   - Simplifies schema without losing functionality

4. KEEP location_history TABLE:
   - Event-based tracking (camera changes) is valuable
   - Not per-frame, so manageable size

5. KEEP features TABLE:
   - Essential for ReID matching
   - Stores feature evolution (mask on/off)
    """)
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
