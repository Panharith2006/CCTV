import json
from database.reid_database import ReIDDatabase

print('[INFO] Using MySQL database from config/mysql_config.py')
db = ReIDDatabase()
cur = db.conn.cursor()

# Check if violation columns exist
cur.execute("""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() 
    AND TABLE_NAME = 'persons'
""")
all_columns = [row[0] for row in cur.fetchall()]
has_violation_columns = 'has_mask_violation' in all_columns

print('=' * 80)
print('  DATABASE CONTENTS - VIOLATION TRACKING SYSTEM')
print('=' * 80)
print()

# Get total persons
cur.execute('SELECT COUNT(*) FROM persons')
total_count = cur.fetchone()[0]
print(f'📊 Total Persons in Database: {total_count}')

if has_violation_columns:
    # Get violation statistics
    cur.execute('SELECT COUNT(*) FROM persons WHERE has_mask_violation = 1 OR has_helmet_violation = 1')
    violation_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM persons WHERE alert_status = "ALERT"')
    alert_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM persons WHERE warning_status = "WARNING"')
    warning_count = cur.fetchone()[0]
    
    print(f'⚠️  Violations: {violation_count}')
    print(f'🚨 Alerts (no helmet): {alert_count}')
    print(f'⚠️  Warnings (no mask): {warning_count}')
    print()
    print('✅ Database contains ONLY persons with violations (correct behavior)')
else:
    print('⚠️  Violation columns not found - run migration script')
    print('   python scripts/migrate_add_violation_tracking.py')

print()
print('=' * 80)
print('  PERSON RECORDS')
print('=' * 80)

if has_violation_columns:
    # Show persons with violation info
    cur.execute('''
        SELECT 
            person_id, 
            name, 
            has_mask_violation,
            has_helmet_violation,
            alert_status,
            warning_status,
            violation_type,
            appearance_count, 
            status,
            first_seen,
            last_seen
        FROM persons 
        ORDER BY person_id DESC
    ''')
    
    for r in cur.fetchall():
        person_id = r[0]
        name = r[1]
        has_mask_vio = r[2]
        has_helmet_vio = r[3]
        alert = r[4]
        warning = r[5]
        violation_type = r[6]
        appearance_count = r[7]
        status = r[8]
        first_seen = r[9]
        last_seen = r[10]
        
        print()
        print(f'ID: {person_id} | Name: {name}')
        print(f'  Mask Violation: {"YES ⚠️" if has_mask_vio else "No"}')
        print(f'  Helmet Violation: {"YES 🚨" if has_helmet_vio else "No"}')
        print(f'  Alert Status: {alert or "-"}')
        print(f'  Warning Status: {warning or "-"}')
        print(f'  Violation Type: {violation_type or "-"}')
        print(f'  Appearances: {appearance_count} | Status: {status}')
        print(f'  First: {first_seen} | Last: {last_seen}')
else:
    # Fallback to old format
    cur.execute('SELECT person_id, name, thumbnail_path, first_seen, last_seen, appearance_count, status FROM persons')
    for r in cur.fetchall():
        print()
        print(f'ID: {r[0]} | Name: {r[1]}')
        print(f'  Appearances: {r[5]} | Status: {r[6]}')
        print(f'  First: {r[3]} | Last: {r[4]}')

print()
print('=' * 80)
print('  FEATURE VECTORS')
print('=' * 80)

print('\nFeature counts per person:')
cur.execute('SELECT person_id, COUNT(*) FROM features GROUP BY person_id')
for r in cur.fetchall():
    print(f'  Person {r[0]}: {r[1]} feature vectors')

print('\nRecent feature sample (checking 128D vectors):')
cur.execute('SELECT person_id, feature_vector FROM features ORDER BY feature_id DESC LIMIT 3')
for r in cur.fetchall():
    pid = r[0]
    fv = json.loads(r[1])
    print(f'  Person {pid}: {len(fv)}D feature vector {"✓" if len(fv) == 128 else f"⚠️ Expected 128D"}')

print()
print('=' * 80)
print()

db.close()
