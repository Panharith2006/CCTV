"""Show final database status after renumbering"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mysql_config import MYSQL_CONFIG
import mysql.connector

conn = mysql.connector.connect(
    host=MYSQL_CONFIG['host'],
    user=MYSQL_CONFIG['user'],
    password=MYSQL_CONFIG['password'],
    database=MYSQL_CONFIG['database']
)
cursor = conn.cursor()

print("\n" + "="*80)
print("  Final Database Status After Renumbering")
print("="*80 + "\n")

# Check all tables
cursor.execute("SELECT MIN(person_id), MAX(person_id), COUNT(*) FROM persons")
p_min, p_max, p_count = cursor.fetchone()

cursor.execute("SELECT MIN(feature_id), MAX(feature_id), COUNT(*) FROM features")
f_min, f_max, f_count = cursor.fetchone()

cursor.execute("SELECT MIN(history_id), MAX(history_id), COUNT(*) FROM location_history")
h_min, h_max, h_count = cursor.fetchone()

print("Table Status:")
print("-" * 80)
print(f"persons:          {p_count} rows, IDs {p_min}-{p_max}, Gaps: {(p_max-p_min+1)-p_count}")
print(f"features:         {f_count} rows, IDs {f_min}-{f_max}, Gaps: {(f_max-f_min+1)-f_count}")
print(f"location_history: {h_count} rows, IDs {h_min}-{h_max}, Gaps: {(h_max-h_min+1)-h_count}")

# Get AUTO_INCREMENT values
print("\nAUTO_INCREMENT Status:")
print("-" * 80)
cursor.execute("""
    SELECT table_name, auto_increment 
    FROM information_schema.tables 
    WHERE table_schema = %s AND auto_increment IS NOT NULL
    ORDER BY table_name
""", (MYSQL_CONFIG['database'],))

for table_name, auto_inc in cursor.fetchall():
    print(f"{table_name:20} -> Next ID: {auto_inc}")

# Show actual person IDs
cursor.execute("SELECT person_id, violation_status, violation_reason FROM persons ORDER BY person_id")
persons = cursor.fetchall()

print("\nActual Person IDs:")
print("-" * 80)
for person_id, v_status, v_reason in persons:
    print(f"  ID {person_id}: Status={v_status}, Reason={v_reason}")

if (p_max - p_min + 1) == p_count and \
   (f_max - f_min + 1) == f_count and \
   (h_max - h_min + 1) == h_count:
    print("\n" + "="*80)
    print("✅ SUCCESS! All IDs are perfectly sequential with ZERO gaps!")
    print("="*80)
    print("\nNext inserts will continue from:")
    print(f"  • persons: ID {p_max + 1}")
    print(f"  • features: ID {f_max + 1}")
    print(f"  • location_history: ID {h_max + 1}")
else:
    print("\n⚠️  Warning: Some gaps still exist")

cursor.close()
conn.close()
