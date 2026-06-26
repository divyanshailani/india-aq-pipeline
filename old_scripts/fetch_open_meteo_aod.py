import os
import sys
import time
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

# Ensure the table exists
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS satellite_aod_features (
    station_id      INTEGER REFERENCES stations(id),
    date            DATE NOT NULL,
    aod_mean        DOUBLE PRECISION,
    aod_max         DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (station_id, date)
);
CREATE INDEX IF NOT EXISTS idx_satellite_aod_lookup ON satellite_aod_features (station_id, date);
"""

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn, cur

def get_target_stations(conn):
    query = """
    SELECT DISTINCT df.station_id, s.latitude, s.longitude
    FROM daily_features df
    JOIN stations s ON df.station_id = s.id
    WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL
    AND df.station_id NOT IN (
        SELECT DISTINCT station_id FROM satellite_aod_features
    )
    """
    return pd.read_sql(query, conn)

def get_date_range(conn):
    query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM daily_features"
    df = pd.read_sql(query, conn)
    return df['min_date'].iloc[0].strftime('%Y-%m-%d'), df['max_date'].iloc[0].strftime('%Y-%m-%d')

def fetch_open_meteo_aod(lat, lon, start_date, end_date):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=aerosol_optical_depth"
    
    attempt = 0
    while True:
        try:
            response = requests.get(url)
            
            if response.status_code == 429:
                attempt += 1
                wait_time = min(60 * attempt, 300) # cap at 5 mins
                print(f"    -> API Limit (429). Backing off for {wait_time} seconds (Attempt {attempt})...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data or "aerosol_optical_depth" not in data["hourly"]:
                return pd.DataFrame()
                
            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                "aod": data["hourly"]["aerosol_optical_depth"]
            })
            
            # Drop nulls
            df = df.dropna(subset=["aod"])
            if df.empty:
                return pd.DataFrame()
                
            # Extract date and aggregate
            df["date"] = df["time"].dt.date
            
            daily_df = df.groupby("date").agg(
                aod_mean=("aod", "mean"),
                aod_max=("aod", "max")
            ).reset_index()
            
            return daily_df
            
        except Exception as e:
            print(f"Error fetching AOD for lat={lat}, lon={lon}: {e}")
            attempt += 1
            if attempt > 10:
                print(f"    -> Giving up after 10 failed network requests.")
                return pd.DataFrame()
            time.sleep(5)

def upsert_aod_data(cur, records):
    if not records:
        return
        
    insert_sql = """
    INSERT INTO satellite_aod_features (station_id, date, aod_mean, aod_max)
    VALUES %s
    ON CONFLICT (station_id, date) DO UPDATE SET
        aod_mean = EXCLUDED.aod_mean,
        aod_max = EXCLUDED.aod_max,
        created_at = now()
    """
    execute_values(cur, insert_sql, records)

def main():
    print("Initializing Open-Meteo Aerosol Optical Depth (AOD) ETL Pipeline...")
    conn, cur = init_db()
    
    stations_df = get_target_stations(conn)
    start_date, end_date = get_date_range(conn)
    
    print(f"Target Date Range: {start_date} to {end_date}")
    print(f"Found {len(stations_df)} distinct stations to process.")
    
    total_stations = len(stations_df)
    for idx, row in stations_df.iterrows():
        station_id = row['station_id']
        lat = row['latitude']
        lon = row['longitude']
        
        print(f"[{idx+1}/{total_stations}] Fetching AOD for Station {station_id} (Lat: {lat:.4f}, Lon: {lon:.4f})...")
        
        daily_aod_df = fetch_open_meteo_aod(lat, lon, start_date, end_date)
        
        if not daily_aod_df.empty:
            # Prepare records for insertion
            records = []
            for _, aod_row in daily_aod_df.iterrows():
                records.append((
                    int(station_id),
                    aod_row['date'],
                    float(aod_row['aod_mean']),
                    float(aod_row['aod_max'])
                ))
            
            upsert_aod_data(cur, records)
            conn.commit()
            print(f"  -> Inserted {len(records)} daily AOD records.")
        else:
            print(f"  -> No valid AOD data found.")
            
        # Respect Open-Meteo free tier rate limits
        time.sleep(1)

    cur.close()
    conn.close()
    print("AOD ETL Pipeline Complete.")

if __name__ == "__main__":
    main()
