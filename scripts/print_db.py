import json
from database.reid_database import ReIDDatabase

print('[INFO] Using MySQL database from config/mysql_config.py')
db = ReIDDatabase()
cur = db.conn.cursor()

print('Persons:')
cur.execute('SELECT person_id, name, thumbnail_path, first_seen, last_seen, appearance_count, status FROM persons')
for r in cur.fetchall():
    print(r)

print('\nFeatures (counts per person):')
cur.execute('SELECT person_id, COUNT(*) FROM features GROUP BY person_id')
for r in cur.fetchall():
    print(r)

print('\nRecent features sample:')
cur.execute('SELECT person_id, feature_vector FROM features ORDER BY feature_id DESC LIMIT 5')
for r in cur.fetchall():
    pid = r[0]
    fv = json.loads(r[1])
    print(pid, len(fv))

db.close()
