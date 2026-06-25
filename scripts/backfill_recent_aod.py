import time
import requests
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from datetime import date, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
START_DATE = "2026-06-21"
END_DATE = "2026-06-25"

def get_stations(conn):
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.date BETWEEN %s AND %s
          AND df.parameter = 'pm25'
    """
    return pd.read_sql(query, conn, params=(START_DATE, END_DATE))

def fetch_open_meteo_aod(lat, lon):
    url = f"{OPEN_METEO_AQ_URL}?latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}&hourly=aerosol_optical_depth"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200: return pd.DataFrame()
            
        data = response.json()
        if "hourly" not in data or "aerosol_optical_depth" not in data["hourly"]:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "aod": data["hourly"]["aerosol_optical_depth"]
        })
        
        df = df.dropna(subset=["aod"])
        if df.empty: return pd.DataFrame()
            
        df["date"] = df["time"].dt.date
        daily_df = df.groupby("date").agg(
            aod_mean=("aod", "mean"),
            aod_max=("aod", "max")
        ).reset_index()
        
        return daily_df
    except Exception as e:
        print(f"Error fetching AOD: {e}")
        return pd.DataFrame()

def upsert_aod_data(cur, records):
    if not records: return
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
    print(f"Backfilling AOD data from {START_DATE} to {END_DATE}...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    stations = get_stations(conn)
    total = len(stations)
    print(f"Found {total} active stations to backfill.")
    
    upserted_count = 0
    for i, row in stations.iterrows():
        sid, lat, lon = row["id"], row["latitude"], row["longitude"]
        if i % 100 == 0:
            print(f"[{i}/{total}] Fetching AOD for station {sid} ({lat:.2f}, {lon:.2f})...")
        
        df = fetch_open_meteo_aod(lat, lon)
        if not df.empty:
            records = [
                (int(sid), aod_row['date'], float(aod_row['aod_mean']), float(aod_row['aod_max']))
                for _, aod_row in df.iterrows()
            ]
            upsert_aod_data(cur, records)
            upserted_count += len(records)
            
        time.sleep(0.1)

    conn.commit()
    print(f"Complete! Upserted {upserted_count} AOD records into satellite_aod_features.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
