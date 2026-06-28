import os
import psycopg2
from psycopg2.extras import RealDictCursor
from src.config import DB_CONFIG

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print("--- Issue #2: Phantom Stations ---")
    cur.execute("SELECT count(*) as total_stations FROM stations")
    total = cur.fetchone()['total_stations']
    cur.execute("""
        SELECT count(*) as phantom 
        FROM stations s 
        LEFT JOIN (SELECT DISTINCT station_id FROM daily_features) f 
        ON s.station_id = f.station_id 
        WHERE f.station_id IS NULL
    """)
    phantom = cur.fetchone()['phantom']
    print(f"Total Stations: {total}, Phantom Stations: {phantom} ({phantom/total*100:.1f}%)")

    print("\n--- Issue #4: Empty Tables ---")
    cur.execute("SELECT count(*) as models FROM model_registry")
    print(f"Rows in model_registry: {cur.fetchone()['models']}")
    cur.execute("SELECT count(*) as preds FROM predictions")
    print(f"Rows in predictions: {cur.fetchone()['preds']}")

    print("\n--- Issue #8: ETL Catchup ---")
    cur.execute("SELECT max(date) as max_date FROM daily_features")
    print(f"Max date in daily_features: {cur.fetchone()['max_date']}")

    cur.close()
    conn.close()
except Exception as e:
    print(f"DB Error: {e}")
