"""
Fetch historical weather from NASA POWER API.
Satellite-based data → global coverage, minimal gaps.

API Docs: https://power.larc.nasa.gov/docs/services/api/

For each station lat/lon, fetches daily:
  - T2M: Temperature at 2m (°C)
  - RH2M: Relative Humidity at 2m (%)
  - WS10M: Wind Speed at 10m (m/s)

Saves to: data/weather_nasa_power.csv
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

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START_DATE = "20210107"   # YYYYMMDD format
END_DATE = "20260528"
SLEEP_BETWEEN_CALLS = 1.0
OUTPUT_CSV = "data/weather_nasa_power.csv"


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


def fetch_nasa_power(lat, lon, start_date, end_date):
    """
    Fetch daily weather from NASA POWER for a single lat/lon.
    Returns DataFrame with: date, temperature, humidity, wind_speed
    """
    params = {
        "parameters": "T2M,RH2M,WS10M",
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
        t2m = params_data.get("T2M", {})
        rh2m = params_data.get("RH2M", {})
        ws10m = params_data.get("WS10M", {})

        rows = []
        for date_str in t2m.keys():
            temp = t2m.get(date_str)
            hum = rh2m.get(date_str)
            wind = ws10m.get(date_str)

            # NASA POWER uses -999 for missing values
            if temp == -999: temp = None
            if hum == -999: hum = None
            if wind == -999: wind = None

            rows.append({
                "date": pd.to_datetime(date_str, format="%Y%m%d").date(),
                "nasa_temperature": temp,
                "nasa_humidity": hum,
                "nasa_wind_speed": wind,
            })

        return pd.DataFrame(rows), "ok"

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            return None, "rate_limited"
        return None, f"http_error: {e}"
    except Exception as e:
        return None, f"error: {e}"


def main():
    print("=" * 60)
    print("🛰️  NASA POWER Weather Fetcher for IndiaAQ")
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

        weather_df, status = fetch_nasa_power(lat, lon, START_DATE, END_DATE)

        if status == "rate_limited":
            print("⏳ Rate limited, waiting 30s...")
            time.sleep(30)
            weather_df, status = fetch_nasa_power(lat, lon, START_DATE, END_DATE)

        if weather_df is not None and not weather_df.empty:
            weather_df["station_id"] = station_id
            null_count = weather_df["nasa_temperature"].isnull().sum()
            total_rows = len(weather_df)
            print(f"✅ {total_rows} days ({null_count} nulls)")
            all_frames.append(weather_df)
        else:
            failed += 1
            print(f"❌ {status}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Combine and save
    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'=' * 60}")
        print(f"✅ Saved {len(result):,} rows to {OUTPUT_CSV}")
        print(f"   Null counts:")
        print(f"   temperature: {result['nasa_temperature'].isnull().sum():,}")
        print(f"   humidity:    {result['nasa_humidity'].isnull().sum():,}")
        print(f"   wind_speed:  {result['nasa_wind_speed'].isnull().sum():,}")
        print(f"❌ Failed: {failed} stations")
        print(f"{'=' * 60}")
    else:
        print("❌ No data fetched!")


if __name__ == "__main__":
    main()
