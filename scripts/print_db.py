import sqlite3, json, os

DB='cctv_reid.db'
if not os.path.exists(DB):
    print('DB not found:', DB)
else:
    conn=sqlite3.connect(DB)
    cur=conn.cursor()
    print('Persons:')
    for r in cur.execute('SELECT person_id, name, thumbnail_path, first_seen, last_seen, appearance_count, status FROM persons').fetchall():
        print(r)
    print('\nFeatures (counts per person):')
    rows=cur.execute('SELECT person_id, COUNT(*) FROM features GROUP BY person_id').fetchall()
    for r in rows:
        print(r)
    print('\nRecent features sample:')
    for r in cur.execute('SELECT person_id, feature_vector FROM features ORDER BY feature_id DESC LIMIT 5').fetchall():
        pid=r[0]
        fv=json.loads(r[1])
        print(pid, len(fv))
    conn.close()