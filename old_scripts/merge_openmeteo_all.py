"""
Merge Open-Meteo historical weather into daily_features for ALL countries.
"""

import psycopg2
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

COUNTRIES = ["GB", "IN", "US", "AU"]

def add_columns_if_missing(conn):
    cols = ["om_temperature", "om_wind_speed", "om_precipitation"]
    with conn.cursor() as cur:
        for c in cols:
            try:
                cur.execute(f"ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS {c} DOUBLE PRECISION;")
            except Exception as e:
                print(f"Error adding {c}: {e}")
                conn.rollback()
    conn.commit()

def merge_country(conn, cc):
    csv_path = f"data/weather_openmeteo_{cc.lower()}.csv"
    if not os.path.exists(csv_path):
        print(f"  ✗ {cc}: File not found: {csv_path}", flush=True)
        return 0
        
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    temp_table = f"temp_om_{cc.lower()}"
    
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
        cur.execute(f"""
            CREATE TEMP TABLE {temp_table} (
                date DATE,
                station_id TEXT,
                om_temperature DOUBLE PRECISION,
                om_wind_speed DOUBLE PRECISION,
                om_precipitation DOUBLE PRECISION
            )
        """)
        
        from psycopg2.extras import execute_values
        data_tuples = list(df[["date", "station_id", "om_temperature", "om_wind_speed", "om_precipitation"]].itertuples(index=False, name=None))
        
        # Batch insert in chunks of 10000
        chunk_size = 10000
        for i in range(0, len(data_tuples), chunk_size):
            chunk = data_tuples[i:i+chunk_size]
            execute_values(cur, f"INSERT INTO {temp_table} VALUES %s", chunk)
        
        cur.execute(f"""
            UPDATE daily_features df
            SET om_temperature = t.om_temperature,
                om_wind_speed = t.om_wind_speed,
                om_precipitation = t.om_precipitation
            FROM {temp_table} t
            WHERE df.station_id = t.station_id::INTEGER
              AND df.date = t.date
              AND df.country_code = '{cc}'
        """)
        
        updated = cur.rowcount
        conn.commit()
        return updated

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    add_columns_if_missing(conn)
    
    total = 0
    for cc in COUNTRIES:
        updated = merge_country(conn, cc)
        print(f"  ✓ {cc}: Updated {updated:,} rows", flush=True)
        total += updated
    
    print(f"\n  TOTAL: {total:,} rows updated with Open-Meteo weather", flush=True)
    conn.close()

if __name__ == "__main__":
    main()
