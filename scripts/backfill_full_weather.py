import time
import requests
import psycopg2
import pandas as pd
from datetime import date, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# We fetch from 2021 up to 7 days ago to avoid the ERA5 historical lag
START_DATE = "2021-01-01"
END_DATE = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")

def get_stations_with_missing_weather(conn):
    """Get stations that have AT LEAST ONE missing weather row in the date range."""
    query = """
        SELECT DISTINCT s.id, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.date BETWEEN %s AND %s
          AND df.om_temperature IS NULL
    """
    return pd.read_sql(query, conn, params=(START_DATE, END_DATE))

def fetch_openmeteo_archive(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
    if resp.status_code != 200:
        print(f"  [!] OpenMeteo error {resp.status_code}: {resp.text}")
        return None
        
    data = resp.json().get("daily", {})
    if not data or not data.get("time"):
        return None
    
    return pd.DataFrame({
        "date": data["time"],
        "om_temperature": data.get("temperature_2m_mean"),
        "om_wind_speed": data.get("wind_speed_10m_max"),
        "om_precipitation": data.get("precipitation_sum"),
        "humidity": data.get("relative_humidity_2m_mean")
    })

def update_daily_features_bulk(conn, station_id, df):
    if df is None or df.empty:
        return 0
        
    # We only want to update rows where the weather is currently NULL
    # to avoid overwriting existing good data.
    sql = """
        UPDATE daily_features
        SET om_temperature = %s,
            om_wind_speed = %s,
            om_precipitation = %s,
            precipitation = COALESCE(precipitation, %s),
            temperature = COALESCE(temperature, %s),
            wind_speed = COALESCE(wind_speed, %s),
            humidity = COALESCE(humidity, %s)
        WHERE station_id = %s 
          AND date = %s 
          AND om_temperature IS NULL
    """
    
    # Filter out rows where om_temperature from API is None (e.g. gaps in API data)
    df_valid = df.dropna(subset=['om_temperature'])
    if df_valid.empty:
        return 0

    values = []
    for _, row in df_valid.iterrows():
        values.append((
            row["om_temperature"], row["om_wind_speed"], row["om_precipitation"],
            row["om_precipitation"], row["om_temperature"], row["om_wind_speed"], row["humidity"],
            station_id, row["date"]
        ))
        
    from psycopg2.extras import execute_batch
    cur = conn.cursor()
    execute_batch(cur, sql, values, page_size=500)
    updated = cur.rowcount
    conn.commit()
    cur.close()
    return updated

def main():
    print(f"🚀 Starting Bulk Weather Backfill (Archive API)")
    print(f"📅 Date Range: {START_DATE} to {END_DATE} (7-day ERA5 Lag Guard)")
    
    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations_with_missing_weather(conn)
    total = len(stations)
    
    print(f"🔍 Found {total} stations with missing weather in this range.")
    if total == 0:
        print("✅ Nothing to do!")
        conn.close()
        return
        
    updated_rows = 0
    start_time = time.time()
    
    for i, row in stations.iterrows():
        sid = int(row["id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        print(f"[{i+1}/{total}] Fetching Station {sid} ({lat:.2f}, {lon:.2f})...", end=" ", flush=True)
        
        try:
            df = fetch_openmeteo_archive(lat, lon)
            if df is not None:
                updated = update_daily_features_bulk(conn, sid, df)
                updated_rows += updated
                print(f"Updated {updated} rows.")
            else:
                print("No data.")
            
            # Respect API limits (OpenMeteo asks for max 10,000 per day, and has minutely limits)
            time.sleep(1.5)
        except Exception as e:
            conn.rollback()
            print(f"Error: {e}")
            time.sleep(2.0)
            
    elapsed = time.time() - start_time
    print(f"\n🎉 Bulk Backfill Complete in {elapsed:.1f}s!")
    print(f"📈 Total rows updated: {updated_rows}")
    print(f"👉 NEXT STEP: Run 'python3 scripts/run_daily_etl.py --recent-days 14' to fill the recent 7-day void!")
    
    conn.close()

if __name__ == "__main__":
    main()
