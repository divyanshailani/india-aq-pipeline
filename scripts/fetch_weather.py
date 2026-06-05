"""
Fetch historical weather data from Open-Meteo API
and update the daily_features table in PostgreSQL.

Open-Meteo: Free, no API key, daily historical weather.
Features: Skips already-fetched stations, retries on rate limit.

Usage: python scripts/fetch_weather.py
"""

import time
import requests
import psycopg2
import pandas as pd

# ── Config ──────────────────────────────────────────────
DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": "5432"
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2021-01-07"
END_DATE = "2026-05-28"
SLEEP_BETWEEN_CALLS = 1.5      # seconds between API calls
MAX_RETRIES = 3                 # retry on rate limit
RETRY_WAIT = 60                 # seconds to wait on 429 error


def get_stations_needing_weather(conn):
    """Get stations that still have NULL weather in daily_features."""
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.parameter = 'pm25'
          AND df.temperature IS NULL
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
        ORDER BY s.id
    """
    return pd.read_sql(query, conn)


def get_total_stations(conn):
    """Get total stations count for progress display."""
    query = """
        SELECT COUNT(DISTINCT station_id) as total
        FROM daily_features WHERE parameter = 'pm25'
    """
    return pd.read_sql(query, conn).iloc[0, 0]


def fetch_weather_for_station(lat, lon, start_date, end_date):
    """Fetch daily weather from Open-Meteo with retry logic."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max",
        "timezone": "Asia/Kolkata"
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)

            # Rate limited — wait and retry
            if resp.status_code == 429:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"\n  ⏳ Rate limited. Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "daily" not in data:
                return None

            daily = data["daily"]
            df = pd.DataFrame({
                "date": daily["time"],
                "temperature": daily.get("temperature_2m_mean"),
                "humidity": daily.get("relative_humidity_2m_mean"),
                "wind_speed": daily.get("wind_speed_10m_max"),
            })
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"\n  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  ⚠ HTTP error: {e}")
            return None
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return None

    print("  ❌ Max retries reached")
    return None


def update_daily_features(conn, station_id, weather_df):
    """Update daily_features with weather data."""
    if weather_df is None or weather_df.empty:
        return 0

    cur = conn.cursor()
    updated = 0

    for _, row in weather_df.iterrows():
        cur.execute("""
            UPDATE daily_features
            SET temperature = %s,
                humidity = %s,
                wind_speed = %s
            WHERE station_id = %s
              AND date = %s
              AND parameter = 'pm25'
              AND temperature IS NULL
        """, (
            row["temperature"],
            row["humidity"],
            row["wind_speed"],
            station_id,
            row["date"]
        ))
        updated += cur.rowcount

    conn.commit()
    return updated


def main():
    print("=" * 60)
    print("🌦  Open-Meteo Weather Fetcher for IndiaAQ")
    print(f"   Date range: {START_DATE} → {END_DATE}")
    print(f"   Sleep: {SLEEP_BETWEEN_CALLS}s | Retry wait: {RETRY_WAIT}s")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)

    # Get total and remaining
    total_all = get_total_stations(conn)
    stations = get_stations_needing_weather(conn)
    remaining = len(stations)
    already_done = total_all - remaining

    print(f"\n📍 Total stations: {total_all}")
    print(f"✅ Already fetched: {already_done}")
    print(f"📥 Remaining: {remaining}\n")

    if remaining == 0:
        print("🎉 All stations already have weather data!")
        conn.close()
        return

    total_updated = 0
    failed = 0

    for idx, row in stations.iterrows():
        station_id = row["id"]
        name = row["name"]
        lat = row["latitude"]
        lon = row["longitude"]
        progress = already_done + idx + 1

        print(f"[{idx+1}/{remaining}] {name[:40]:<40} ({lat:.2f}, {lon:.2f})", end="  ")

        weather_df = fetch_weather_for_station(lat, lon, START_DATE, END_DATE)

        if weather_df is not None:
            count = update_daily_features(conn, station_id, weather_df)
            total_updated += count
            print(f"✅ {count} rows")
        else:
            failed += 1
            print("❌ failed")

        time.sleep(SLEEP_BETWEEN_CALLS)

    conn.close()

    print("\n" + "=" * 60)
    print(f"✅ Done! {total_updated:,} rows updated")
    print(f"✅ Succeeded: {remaining - failed} stations")
    print(f"❌ Failed: {failed} stations")
    print("=" * 60)


if __name__ == "__main__":
    main()
