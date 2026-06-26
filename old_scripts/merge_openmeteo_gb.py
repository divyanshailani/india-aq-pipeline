"""
Merge Open-Meteo GB historical weather into daily_features.
"""

import psycopg2
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

INPUT_CSV = "data/weather_openmeteo_gb.csv"

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

def merge_data(conn):
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return
        
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # We will use fast batch UPDATE via temp table
    temp_table = "temp_om_gb"
    
    with conn.cursor() as cur:
        # Create temp table
        cur.execute(f"""
            CREATE TEMP TABLE {temp_table} (
                date DATE,
                station_id TEXT,
                om_temperature DOUBLE PRECISION,
                om_wind_speed DOUBLE PRECISION,
                om_precipitation DOUBLE PRECISION
            )
        """)
        
        # Insert into temp table
        from psycopg2.extras import execute_values
        data_tuples = list(df[["date", "station_id", "om_temperature", "om_wind_speed", "om_precipitation"]].itertuples(index=False, name=None))
        execute_values(cur, f"INSERT INTO {temp_table} VALUES %s", data_tuples)
        
        # Update daily_features
        cur.execute(f"""
            UPDATE daily_features df
            SET om_temperature = t.om_temperature,
                om_wind_speed = t.om_wind_speed,
                om_precipitation = t.om_precipitation
            FROM {temp_table} t
            WHERE df.station_id = t.station_id::INTEGER
              AND df.date = t.date
              AND df.country_code = 'GB'
        """)
        
        updated = cur.rowcount
        conn.commit()
        print(f"Successfully updated {updated} rows in daily_features with Open-Meteo data.")

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    add_columns_if_missing(conn)
    merge_data(conn)
    conn.close()
