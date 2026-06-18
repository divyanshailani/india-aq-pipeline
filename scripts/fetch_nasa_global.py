"""
Fetch NASA POWER weather for all countries — DAILY MODE.
=========================================================
For admin panel daily runs: fetches only the last N days
of weather data (default: 14 days) instead of the full 5-year
historical range.

For full historical backfill, use:
    python scripts/fetch_nasa_global.py --backfill
"""

import sys
import os
import time
import argparse
from datetime import datetime, timedelta

import requests
import psycopg2
import pandas as pd

# Import shared config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG, DATA_DIR

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
SLEEP = 0.8


def get_stations_for_country(conn, country_code, limit=None):
    """Get stations with PM2.5 features for a country."""
    query = """
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.parameter = 'pm25'
          AND s.country_code = %s
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
        ORDER BY s.id
    """
    if limit:
        query += f" LIMIT {limit}"
    return pd.read_sql(query, conn, params=(country_code,))


def fetch_nasa_power(lat, lon, start_date, end_date):
    """Fetch daily weather including precip and wind direction."""
    params = {
        "parameters": "T2M,RH2M,WS10M,PRECTOTCORR,WD10M",
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

        p = data["properties"]["parameter"]
        t2m = p.get("T2M", {})
        rh2m = p.get("RH2M", {})
        ws10m = p.get("WS10M", {})
        precip = p.get("PRECTOTCORR", {})
        wd10m = p.get("WD10M", {})

        rows = []
        for ds in t2m.keys():
            def clean(v):
                return None if v == -999 else v

            rows.append({
                "date": pd.to_datetime(ds, format="%Y%m%d").date(),
                "nasa_temperature": clean(t2m.get(ds)),
                "nasa_humidity": clean(rh2m.get(ds)),
                "nasa_wind_speed": clean(ws10m.get(ds)),
                "precipitation": clean(precip.get(ds)),
                "wind_direction": clean(wd10m.get(ds)),
            })

        return pd.DataFrame(rows), "ok"

    except requests.exceptions.HTTPError:
        return None, f"http_{resp.status_code}"
    except Exception as e:
        return None, f"error: {e}"


def fetch_country(country_code, start_date, end_date, max_stations=None):
    """Fetch weather for all stations in a country."""
    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations_for_country(conn, country_code, limit=max_stations)
    conn.close()

    total = len(stations)
    print(f"\n{'=' * 60}")
    print(f"  {country_code}: {total} stations | {start_date} → {end_date}")
    print(f"{'=' * 60}")

    if total == 0:
        print("  No stations found!")
        return None

    all_frames = []
    failed = 0

    for idx, row in stations.iterrows():
        sid = row["id"]
        name = row["name"][:35]
        lat, lon = row["latitude"], row["longitude"]

        print(f"  [{idx+1}/{total}] {name:<35} ({lat:.2f}, {lon:.2f})", end="  ")

        weather_df, status = fetch_nasa_power(lat, lon, start_date, end_date)

        if status == "rate_limited":
            print("⏳ Rate limited, waiting 30s...")
            time.sleep(30)
            weather_df, status = fetch_nasa_power(lat, lon, start_date, end_date)

        if weather_df is not None and not weather_df.empty:
            weather_df["station_id"] = sid
            print(f"✅ {len(weather_df)} days")
            all_frames.append(weather_df)
        else:
            failed += 1
            print(f"❌ {status}")

        time.sleep(SLEEP)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        output = os.path.join(DATA_DIR, f"weather_nasa_{country_code.lower()}.csv")
        result.to_csv(output, index=False)
        print(f"\n  ✅ Saved {len(result):,} rows to {output}")
        print(f"  ❌ Failed: {failed} stations")
        return result
    else:
        print("  ❌ No data fetched!")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch NASA POWER weather")
    parser.add_argument("--backfill", action="store_true",
                        help="Full 5-year historical fetch (slow)")
    parser.add_argument("--days", type=int, default=14,
                        help="Number of recent days to fetch (default: 14)")
    parser.add_argument("--max-stations", type=int, default=None,
                        help="Max stations per country (for testing)")
    args = parser.parse_args()

    if args.backfill:
        start = "20210107"
        end = datetime.now().strftime("%Y%m%d")
        max_st = args.max_stations
        print("🔄 BACKFILL MODE: Fetching full historical range")
    else:
        end_dt = datetime.now() - timedelta(days=1)
        start_dt = end_dt - timedelta(days=args.days)
        start = start_dt.strftime("%Y%m%d")
        end = end_dt.strftime("%Y%m%d")
        max_st = args.max_stations or 50  # limit for daily mode
        print(f"📡 DAILY MODE: Last {args.days} days, max {max_st} stations/country")

    for cc in ["US", "GB", "AU"]:
        fetch_country(cc, start, end, max_stations=max_st)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
