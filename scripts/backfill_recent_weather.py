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

def fetch_openmeteo(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=15)
    if resp.status_code != 200: return None
    data = resp.json().get("daily", {})
    if not data: return None
    
    return pd.DataFrame({
        "date": data["time"],
        "om_temperature": data.get("temperature_2m_mean"),
        "om_wind_speed": data.get("wind_speed_10m_max"),
        "om_precipitation": data.get("precipitation_sum"),
        "humidity": data.get("relative_humidity_2m_mean")
    })

def update_daily_features(conn, station_id, df):
    if df is None or df.empty: return 0
    cur = conn.cursor()
    updated = 0
    for _, row in df.iterrows():
        # Update OM fields and standard fields used by the model
        cur.execute("""
            UPDATE daily_features
            SET om_temperature = %s,
                om_wind_speed = %s,
                om_precipitation = %s,
                precipitation = COALESCE(precipitation, %s),
                temperature = COALESCE(temperature, %s),
                wind_speed = COALESCE(wind_speed, %s),
                humidity = COALESCE(humidity, %s)
            WHERE station_id = %s AND date = %s AND parameter = 'pm25'
        """, (
            row["om_temperature"], row["om_wind_speed"], row["om_precipitation"],
            row["om_precipitation"], row["om_temperature"], row["om_wind_speed"], row["humidity"],
            station_id, row["date"]
        ))
        updated += cur.rowcount
    conn.commit()
    return updated

def main():
    print(f"Backfilling weather data from {START_DATE} to {END_DATE}...")
    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations(conn)
    total = len(stations)
    print(f"Found {total} active stations to backfill.")
    
    updated_rows = 0
    for i, row in stations.iterrows():
        sid, lat, lon = row["id"], row["latitude"], row["longitude"]
        if i % 100 == 0:
            print(f"[{i}/{total}] Fetching for station {sid} ({lat:.2f}, {lon:.2f})...")
        
        try:
            df = fetch_openmeteo(lat, lon)
            updated_rows += update_daily_features(conn, sid, df)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching station {sid}: {e}")
            
    print(f"Complete! Updated {updated_rows} rows in daily_features.")
    conn.close()

if __name__ == "__main__":
    main()
