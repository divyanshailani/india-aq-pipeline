"""
Fetch historical weather from Visual Crossing API.
Ground station + model data → good India coverage.

Free tier: 1000 records/day
Sign up: https://www.visualcrossing.com/sign-up

For each station lat/lon, fetches daily:
  - temp: Temperature (°C)
  - humidity: Relative Humidity (%)
  - windspeed: Wind Speed (km/h → convert to m/s)

Saves to: data/weather_visual_crossing.csv
"""

import time
import requests
import psycopg2
import pandas as pd

# ── Config ──────────────────────────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

# ⚠️ REPLACE WITH YOUR FREE API KEY from visualcrossing.com
VC_API_KEY = "YOUR_API_KEY_HERE"

VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
START_DATE = "2021-01-07"
END_DATE = "2026-05-28"
SLEEP_BETWEEN_CALLS = 1.5
OUTPUT_CSV = "data/weather_visual_crossing.csv"


def get_stations(conn):
    """Get all stations with PM2.5 data."""
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.parameter = 'pm25'
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
        ORDER BY s.id
    """
    return pd.read_sql(query, conn)


def fetch_visual_crossing(lat, lon, start_date, end_date, api_key):
    """
    Fetch daily weather from Visual Crossing for a single lat/lon.
    Returns DataFrame with: date, temperature, humidity, wind_speed
    """
    url = f"{VC_BASE_URL}/{lat},{lon}/{start_date}/{end_date}"
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": api_key,
        "contentType": "json",
        "elements": "datetime,temp,humidity,windspeed"
    }

    try:
        resp = requests.get(url, params=params, timeout=60)

        if resp.status_code == 429:
            return None, "rate_limited"
        if resp.status_code == 401:
            return None, "invalid_api_key"

        resp.raise_for_status()
        data = resp.json()

        if "days" not in data:
            return None, "no_data"

        rows = []
        for day in data["days"]:
            wind_ms = day.get("windspeed")
            if wind_ms is not None:
                wind_ms = wind_ms / 3.6  # km/h → m/s

            rows.append({
                "date": day["datetime"],
                "vc_temperature": day.get("temp"),
                "vc_humidity": day.get("humidity"),
                "vc_wind_speed": wind_ms,
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df, "ok"

    except Exception as e:
        return None, f"error: {e}"


def main():
    if VC_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  Set your Visual Crossing API key first!")
        print("   Sign up free: https://www.visualcrossing.com/sign-up")
        print("   Then edit VC_API_KEY in this script.")
        return

    print("=" * 60)
    print("🌤️  Visual Crossing Weather Fetcher for IndiaAQ")
    print(f"   Date range: {START_DATE} → {END_DATE}")
    print(f"   Output: {OUTPUT_CSV}")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations(conn)
    total = len(stations)
    print(f"\n📍 {total} stations to fetch\n")
    conn.close()

    all_frames = []
    failed = 0

    for idx, row in stations.iterrows():
        station_id = row["id"]
        name = row["name"]
        lat = row["latitude"]
        lon = row["longitude"]

        print(f"[{idx+1}/{total}] {name[:40]:<40} ({lat:.2f}, {lon:.2f})", end="  ")

        weather_df, status = fetch_visual_crossing(lat, lon, START_DATE, END_DATE, VC_API_KEY)

        if status == "rate_limited":
            print("⏳ Rate limited, waiting 60s...")
            time.sleep(60)
            weather_df, status = fetch_visual_crossing(lat, lon, START_DATE, END_DATE, VC_API_KEY)

        if weather_df is not None and not weather_df.empty:
            weather_df["station_id"] = station_id
            null_count = weather_df["vc_temperature"].isnull().sum()
            total_rows = len(weather_df)
            print(f"✅ {total_rows} days ({null_count} nulls)")
            all_frames.append(weather_df)
        else:
            failed += 1
            print(f"❌ {status}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'=' * 60}")
        print(f"✅ Saved {len(result):,} rows to {OUTPUT_CSV}")
        print(f"   Null counts:")
        print(f"   temperature: {result['vc_temperature'].isnull().sum():,}")
        print(f"   humidity:    {result['vc_humidity'].isnull().sum():,}")
        print(f"   wind_speed:  {result['vc_wind_speed'].isnull().sum():,}")
        print(f"❌ Failed: {failed} stations")
        print(f"{'=' * 60}")
    else:
        print("❌ No data fetched!")


if __name__ == "__main__":
    main()
