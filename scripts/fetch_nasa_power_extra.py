"""
Fetch additional weather parameters from NASA POWER:
  - PRECTOTCORR: Precipitation (mm/day) — rain washes out PM2.5
  - WD10M: Wind Direction at 10m (degrees) — where pollution comes from

Saves to: data/weather_nasa_power_extra.csv
Then use update script to push into DB.
"""

import time
import requests
import psycopg2
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START_DATE = "20210107"
END_DATE = "20260528"
SLEEP_BETWEEN_CALLS = 1.0
OUTPUT_CSV = "data/weather_nasa_power_extra.csv"


def get_stations(conn):
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.parameter = 'pm25'
          AND s.latitude IS NOT NULL
        ORDER BY s.id
    """
    return pd.read_sql(query, conn)


def fetch_extra_params(lat, lon, start_date, end_date):
    params = {
        "parameters": "PRECTOTCORR,WD10M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    try:
        resp = requests.get(NASA_POWER_URL, params=params, timeout=60)
        if resp.status_code == 429:
            return None, "rate_limited"
        resp.raise_for_status()
        data = resp.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            return None, "no_data"

        params_data = data["properties"]["parameter"]
        precip = params_data.get("PRECTOTCORR", {})
        wind_dir = params_data.get("WD10M", {})

        rows = []
        for date_str in precip.keys():
            p = precip.get(date_str)
            wd = wind_dir.get(date_str)
            if p == -999: p = None
            if wd == -999: wd = None

            rows.append({
                "date": pd.to_datetime(date_str, format="%Y%m%d").date(),
                "precipitation": p,
                "wind_direction": wd,
            })

        return pd.DataFrame(rows), "ok"

    except Exception as e:
        return None, f"error: {e}"


def main():
    print("=" * 60)
    print("🌧️  NASA POWER — Precipitation & Wind Direction")
    print(f"   Output: {OUTPUT_CSV}")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations(conn)
    total = len(stations)
    print(f"\n📍 {total} stations\n")
    conn.close()

    all_frames = []
    failed = 0

    for idx, row in stations.iterrows():
        station_id = row["id"]
        name = row["name"]
        lat, lon = row["latitude"], row["longitude"]

        print(f"[{idx+1}/{total}] {name[:35]:<35} ({lat:.2f}, {lon:.2f})", end="  ")

        df, status = fetch_extra_params(lat, lon, START_DATE, END_DATE)

        if status == "rate_limited":
            print("⏳ Rate limited, waiting 30s...")
            time.sleep(30)
            df, status = fetch_extra_params(lat, lon, START_DATE, END_DATE)

        if df is not None and not df.empty:
            df["station_id"] = station_id
            nulls = df["precipitation"].isnull().sum()
            print(f"✅ {len(df)} days ({nulls} nulls)")
            all_frames.append(df)
        else:
            failed += 1
            print(f"❌ {status}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'=' * 60}")
        print(f"✅ Saved {len(result):,} rows to {OUTPUT_CSV}")
        print(f"   precipitation nulls: {result['precipitation'].isnull().sum():,}")
        print(f"   wind_direction nulls: {result['wind_direction'].isnull().sum():,}")
        print(f"❌ Failed: {failed}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
