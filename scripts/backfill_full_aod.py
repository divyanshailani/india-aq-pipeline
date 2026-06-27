import time
import requests
import psycopg2
import pandas as pd
from datetime import date, timedelta
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
START_DATE = "2021-01-01"
END_DATE = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")

def get_stations_with_missing_aod(conn):
    query = """
        SELECT DISTINCT s.id, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.date BETWEEN %s AND %s
          AND df.om_aerosol_optical_depth IS NULL
    """
    return pd.read_sql(query, conn, params=(START_DATE, END_DATE))


def fetch_openmeteo_aod(lat, lon):
    all_dfs = []
    # Chunk by year to prevent Open-Meteo API timeouts on massive 5-year requests
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    
    for year in years:
        start = f"{year}-01-01"
        if year == 2026:
            end = END_DATE
            if start > end: continue
        else:
            end = f"{year}-12-31"
            
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ["aerosol_optical_depth"],
            "timezone": "auto"
        }
        resp = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=30)
        
        # Sleep to avoid hitting 100 req/min Open-Meteo limit (5 workers * 6 years = massive burst)
        time.sleep(2.0)
        
        if resp.status_code != 200:
            if resp.status_code == 429:
                raise Exception("Rate Limited (429)")
            # Ignore other errors like 400 for future dates
            continue
            
        data = resp.json().get("hourly", {})
        aod_array = data.get("aerosol_optical_depth", [])
        time_array = data.get("time", [])
        
        if not aod_array or not time_array:
            continue
            
        df_hourly = pd.DataFrame({
            "time": pd.to_datetime(time_array),
            "aod": aod_array
        })
        
        df_hourly['date'] = df_hourly['time'].dt.date
        df_daily = df_hourly.groupby('date')['aod'].mean().reset_index()
        df_daily.rename(columns={"aod": "om_aerosol_optical_depth"}, inplace=True)
        all_dfs.append(df_daily)
        
        # Micro-sleep between chunks for safety
        time.sleep(0.2)
        
    if not all_dfs:
        return None
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df
def update_aod_bulk(conn, station_id, df):
    if df is None or df.empty:
        return 0
        
    sql = """
        UPDATE daily_features
        SET om_aerosol_optical_depth = %s
        WHERE station_id = %s 
          AND date = %s 
          AND om_aerosol_optical_depth IS NULL
    """
    
    df_valid = df.dropna(subset=['om_aerosol_optical_depth'])
    if df_valid.empty:
        return 0

    values = []
    for _, row in df_valid.iterrows():
        values.append((
            row["om_aerosol_optical_depth"],
            station_id, 
            row["date"]
        ))
        
    from psycopg2.extras import execute_batch
    cur = conn.cursor()
    execute_batch(cur, sql, values, page_size=500)
    updated = len(values)
    conn.commit()
    cur.close()
    return updated

def process_station(row, total, index):
    sid = int(row["id"])
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    
    conn = psycopg2.connect(**DB_CONFIG)
    updated = 0
    success = False
    
    # Simple retry logic for 429s
    for attempt in range(3):
        try:
            df = fetch_openmeteo_aod(lat, lon)
            if df is not None:
                updated = update_aod_bulk(conn, sid, df)
            success = True
            break
        except Exception as e:
            conn.rollback()
            if "429" in str(e):
                print(f"⚠️ Rate limit hit on station {sid}. Waiting 30s...")
                time.sleep(30) # Wait 30 seconds on rate limit
            else:
                print(f"Error on station {sid}: {e}")
                break
                
    conn.close()
    return sid, updated, success

def main():
    print(f"🚀 Starting Bulk AOD Backfill")
    print(f"📅 Date Range: {START_DATE} to {END_DATE}")
    
    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations_with_missing_aod(conn)
    conn.close()
    
    total = len(stations)
    print(f"🔍 Found {total} stations with missing AOD.")
    if total == 0:
        print("✅ Nothing to do!")
        return
        
    updated_rows = 0
    start_time = time.time()
    
    # Use 1 thread to avoid aggressive rate limiting
    MAX_WORKERS = 1
    print(f"⚡ Using {MAX_WORKERS} parallel workers to respect API limits...")
    
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_station, row, total, i): i for i, row in stations.iterrows()}
        
        for future in as_completed(futures):
            completed += 1
            sid, updated, success = future.result()
            
            if success:
                print(f"[{completed}/{total}] Station {sid} completed. Updated {updated} AOD rows.")
            else:
                print(f"[{completed}/{total}] Station {sid} FAILED.")
                
            # Add a small delay between tasks overall to reduce continuous burst pressure
            time.sleep(1)
            
    elapsed = time.time() - start_time
    print(f"\n🎉 Bulk AOD Backfill Complete in {elapsed:.1f}s!")
    print(f"📈 Total rows updated: {updated_rows}")
    print(f"👉 NEXT STEP: NOW you can safely run 'python3 scripts/run_daily_etl.py --recent-days 14'!")

if __name__ == "__main__":
    main()
