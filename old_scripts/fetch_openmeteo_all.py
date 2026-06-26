"""
Fetch historical weather from Open-Meteo Archive API for ALL countries.
Uses ERA5 reanalysis — same physics engine as their forecast API (Zero Source Shift).
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
COUNTRIES = ["IN", "US", "AU"]  # GB already done

def get_stations(conn, country_code):
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.country_code = %s
          AND df.parameter = 'pm25'
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
        ORDER BY s.id
    """
    return pd.read_sql(query, conn, params=(country_code,))

def fetch_openmeteo(lat, lon, retries=3):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"],
        "timezone": "auto"
    }
    for attempt in range(retries):
        try:
            resp = requests.get(URL, params=params, timeout=15)
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
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    
    for cc in COUNTRIES:
        output_csv = f"data/weather_openmeteo_{cc.lower()}.csv"
        stations = get_stations(conn, cc)
        
        # Resume: check which stations are already fetched
        done_ids = set()
        if os.path.exists(output_csv):
            existing = pd.read_csv(output_csv)
            done_ids = set(existing["station_id"].unique())
            print(f"\n  Resuming {cc}: {len(done_ids)} stations already done", flush=True)
        
        remaining = stations[~stations["id"].isin(done_ids)]
        print(f"\n{'='*60}", flush=True)
        print(f"  {cc}: {len(remaining)} stations remaining (of {len(stations)} total)", flush=True)
        print(f"{'='*60}", flush=True)
        
        if len(remaining) == 0:
            print(f"  ✓ {cc}: All stations already fetched!", flush=True)
            continue
        
        all_data = []
        errors = 0
        
        for i, row in remaining.iterrows():
            sid = row["id"]
            lat = row["latitude"]
            lon = row["longitude"]
            
            idx = len(done_ids) + len(all_data) + 1
            if idx % 50 == 0 or len(all_data) == 0 or (len(all_data) + 1) == len(remaining):
                print(f"  [{idx}/{len(stations)}] {sid} ({lat:.2f}, {lon:.2f})...", flush=True)
            
            try:
                df = fetch_openmeteo(lat, lon)
                if not df.empty:
                    df["station_id"] = sid
                    all_data.append(df)
                time.sleep(0.3)
            except Exception as e:
                errors += 1
                print(f"  ERROR {sid}: {e}", flush=True)
                time.sleep(2)
        
        # Append to existing CSV or create new
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            if os.path.exists(output_csv):
                existing = pd.read_csv(output_csv)
                final_df = pd.concat([existing, new_df], ignore_index=True)
            else:
                final_df = new_df
            final_df.to_csv(output_csv, index=False)
            nulls = final_df["om_temperature"].isna().sum()
            print(f"\n  ✓ {cc}: Saved {len(final_df):,} rows to {output_csv}", flush=True)
            print(f"    Nulls: {nulls}/{len(final_df)} ({nulls/len(final_df)*100:.2f}%)", flush=True)
            print(f"    Errors: {errors}", flush=True)
    
    conn.close()
    print(f"\n{'='*60}", flush=True)
    print(f"  ALL COUNTRIES COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
