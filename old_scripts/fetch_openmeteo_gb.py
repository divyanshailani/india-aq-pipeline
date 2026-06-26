"""
Fetch historical weather from Open-Meteo Archive API for GB Pilot.
"""

import time
import requests
import psycopg2
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2021-01-01"
END_DATE = "2026-06-16"
OUTPUT_CSV = "data/weather_openmeteo_gb.csv"

def get_gb_stations(conn):
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.country_code = 'GB'
          AND df.parameter = 'pm25'
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
        ORDER BY s.id
    """
    return pd.read_sql(query, conn)

def fetch_openmeteo(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"],
        "timezone": "auto"
    }
    resp = requests.get(URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    daily = data.get("daily", {})
    if not daily:
        return pd.DataFrame()
        
    df = pd.DataFrame({
        "date": daily["time"],
        "om_temperature": daily["temperature_2m_mean"],
        "om_wind_speed": daily["wind_speed_10m_max"],
        "om_precipitation": daily["precipitation_sum"]
    })
    return df

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_gb_stations(conn)
    print(f"Found {len(stations)} GB stations.")
    
    all_data = []
    
    for i, row in stations.iterrows():
        sid = row["id"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        print(f"[{i+1}/{len(stations)}] Fetching for {sid} ({lat:.2f}, {lon:.2f})...", flush=True)
        try:
            df = fetch_openmeteo(lat, lon)
            if not df.empty:
                df["station_id"] = sid
                all_data.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f" Error fetching {sid}: {e}", flush=True)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(final_df)} rows to {OUTPUT_CSV}", flush=True)
    else:
        print("No data fetched.", flush=True)

if __name__ == "__main__":
    main()
