"""
Fetch fire hotspot data from NASA FIRMS API.
Satellite-detected active fires → regional_fire_count per station per day.

API: https://firms.modaps.eosdis.nasa.gov/api/
Register for MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/area/

Strategy:
  1. Fetch all fire points in India by date chunks
  2. For each station-day, count fires within 100km radius
  3. Save as CSV with: station_id, date, fire_count

Uses VIIRS_SNPP_SP (standard product) for archive data.
"""

import time
import math
import requests
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

DB_CONFIG = {
    "dbname": "indiaaq", "user": "postgres",
    "password": "8765", "host": "localhost", "port": "5432"
}

# ⚠️ Get your free MAP_KEY from: https://firms.modaps.eosdis.nasa.gov/api/area/
FIRMS_MAP_KEY = "YOUR_MAP_KEY_HERE"

# India bounding box (approximate)
INDIA_BBOX = "68,6,97,37"   # west,south,east,north

RADIUS_KM = 100  # count fires within this radius of each station
OUTPUT_CSV = "data/fire_counts_firms.csv"

# Date range
START_DATE = "2021-01-07"
END_DATE = "2026-05-28"


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


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


def fetch_firms_chunk(map_key, bbox, date_str, ndays=10):
    """
    Fetch fire data for a bounding box and date range.
    FIRMS API allows max 10 days per request for archive.
    """
    # Use archive endpoint for past data
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_SP/{bbox}/{ndays}/{date_str}"

    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 429:
            return None, "rate_limited"
        if resp.status_code == 401:
            return None, "invalid_key"
        resp.raise_for_status()

        if not resp.text.strip() or "Error" in resp.text[:100]:
            return pd.DataFrame(), "no_data"

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        if df.empty:
            return df, "ok"

        # Keep relevant columns
        if 'acq_date' in df.columns:
            df['date'] = pd.to_datetime(df['acq_date']).dt.date
        elif 'ACQ_DATE' in df.columns:
            df['date'] = pd.to_datetime(df['ACQ_DATE']).dt.date

        return df, "ok"

    except Exception as e:
        return None, f"error: {e}"


def count_fires_near_station(fire_df, station_lat, station_lon, radius_km):
    """Count fires within radius_km of a station, grouped by date."""
    if fire_df.empty:
        return {}

    lat_col = 'latitude' if 'latitude' in fire_df.columns else 'LATITUDE'
    lon_col = 'longitude' if 'longitude' in fire_df.columns else 'LONGITUDE'

    distances = fire_df.apply(
        lambda row: haversine_km(station_lat, station_lon,
                                  row[lat_col], row[lon_col]),
        axis=1
    )

    nearby = fire_df[distances <= radius_km]
    return nearby.groupby('date').size().to_dict()


def main():
    if FIRMS_MAP_KEY == "YOUR_MAP_KEY_HERE":
        print("⚠️  Get your free FIRMS MAP_KEY:")
        print("   https://firms.modaps.eosdis.nasa.gov/api/area/")
        print("   Then edit FIRMS_MAP_KEY in this script.")
        return

    print("=" * 60)
    print("🔥 NASA FIRMS Fire Data Fetcher")
    print(f"   Radius: {RADIUS_KM}km | Output: {OUTPUT_CSV}")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations(conn)
    print(f"📍 {len(stations)} stations")
    conn.close()

    # Fetch fire data in 10-day chunks
    print("\n📡 Fetching fire data for India...")
    all_fires = []

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    current = start
    chunk_num = 0

    while current < end:
        chunk_end = min(current + timedelta(days=9), end)
        date_str = current.strftime("%Y-%m-%d")
        ndays = (chunk_end - current).days + 1

        chunk_num += 1
        print(f"  Chunk {chunk_num}: {date_str} ({ndays} days)", end="  ")

        fires, status = fetch_firms_chunk(FIRMS_MAP_KEY, INDIA_BBOX, date_str, ndays)

        if status == "rate_limited":
            print("⏳ Rate limited, waiting 60s...")
            time.sleep(60)
            fires, status = fetch_firms_chunk(FIRMS_MAP_KEY, INDIA_BBOX, date_str, ndays)

        if fires is not None and not fires.empty:
            all_fires.append(fires)
            print(f"✅ {len(fires)} fires")
        elif fires is not None:
            print(f"✅ 0 fires")
        else:
            print(f"❌ {status}")

        current = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not all_fires:
        print("❌ No fire data fetched!")
        return

    all_fire_df = pd.concat(all_fires, ignore_index=True)
    print(f"\n🔥 Total fire points: {len(all_fire_df):,}")

    # Count fires per station per day
    print("\n📊 Counting fires near each station...")
    results = []

    for idx, srow in stations.iterrows():
        station_id = srow['id']
        counts = count_fires_near_station(
            all_fire_df, srow['latitude'], srow['longitude'], RADIUS_KM
        )
        for date, count in counts.items():
            results.append({
                'station_id': station_id,
                'date': date,
                'fire_count': count
            })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(stations)}] stations processed...")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'=' * 60}")
    print(f"✅ Saved {len(result_df):,} rows to {OUTPUT_CSV}")
    print(f"   Date range: {result_df['date'].min()} → {result_df['date'].max()}")
    print(f"   Mean fire count: {result_df['fire_count'].mean():.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
