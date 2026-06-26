import sys, psycopg2
sys.path.insert(0, '.')
from src.config import DB_CONFIG
import src.features as features_module

print("Running fast ETL...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Insert new rows that pass basic quality checks (last 14 days)
cur.execute("""
    INSERT INTO clean_measurements
        (station_id, sensor_id, parameter, value, unit,
         datetime_utc, datetime_local, cleaning_flags, is_valid)
    SELECT r.station_id, r.sensor_id, r.parameter, r.value, r.unit,
           r.datetime_utc, r.datetime_local,
           ARRAY['auto_cleaned']::text[], true
    FROM raw_measurements r
    WHERE NOT EXISTS (
        SELECT 1 FROM clean_measurements c
        WHERE c.station_id = r.station_id
          AND c.parameter = r.parameter
          AND c.datetime_utc = r.datetime_utc
    )
    AND r.datetime_utc >= NOW() - INTERVAL '14 days'
    AND r.value > 0
    AND r.value NOT IN (999.99, 9999, -999)
""")
inserted = cur.rowcount
print(f"Inserted {inserted} rows into clean_measurements.")
conn.commit()

# Rebuild features for affected stations
cur.execute("""
    SELECT DISTINCT station_id FROM raw_measurements
    WHERE datetime_utc >= NOW() - INTERVAL '14 days'
""")
affected_stations = [row[0] for row in cur.fetchall()]
print(f"Rebuilding features for {len(affected_stations)} stations...")
conn.close()

conn = psycopg2.connect(**DB_CONFIG)
features_summary = features_module.run_feature_pipeline(conn, station_ids=affected_stations)
print(features_summary)
conn.close()
