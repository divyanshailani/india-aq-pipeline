"""
Process NASA FIRMS fire data into regional_fire_count per station per day.
For each station, counts fires within 100km radius for each day.

Uses vectorized numpy for speed (3.6M fires × 478 stations).
"""

import math
import numpy as np
import pandas as pd
import psycopg2

DB_CONFIG = {
    "dbname": "indiaaq", "user": "postgres",
    "password": "8765", "host": "localhost", "port": "5432"
}

FIRE_ARCHIVE = "data/raw/DL_FIRE_SV-C2_758885/fire_archive_SV-C2_758885.csv"
FIRE_NRT = "data/raw/DL_FIRE_SV-C2_758885/fire_nrt_SV-C2_758885.csv"
RADIUS_KM = 100
OUTPUT_CSV = "data/fire_counts_firms.csv"


def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized haversine: one point vs array of points. Returns km."""
    R = 6371
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2_arr)
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


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


def main():
    print("=" * 60)
    print("🔥 Processing FIRMS Fire Data → Regional Fire Counts")
    print(f"   Radius: {RADIUS_KM}km | Output: {OUTPUT_CSV}")
    print("=" * 60)

    # Load fire data
    print("\n📂 Loading fire CSVs...")
    fires_archive = pd.read_csv(FIRE_ARCHIVE, usecols=['latitude', 'longitude', 'acq_date', 'confidence'])
    fires_nrt = pd.read_csv(FIRE_NRT, usecols=['latitude', 'longitude', 'acq_date', 'confidence'])

    fires = pd.concat([fires_archive, fires_nrt], ignore_index=True)
    fires['acq_date'] = pd.to_datetime(fires['acq_date']).dt.date

    # Filter: only nominal/high confidence fires
    # confidence: 'l'=low, 'n'=nominal, 'h'=high (for VIIRS)
    fires = fires[fires['confidence'].isin(['n', 'h'])]

    print(f"   Total fire points: {len(fires):,}")
    print(f"   Date range: {fires['acq_date'].min()} → {fires['acq_date'].max()}")

    # Pre-compute fire arrays for speed
    fire_lats = fires['latitude'].values
    fire_lons = fires['longitude'].values
    fire_dates = fires['acq_date'].values

    # Load stations
    conn = psycopg2.connect(**DB_CONFIG)
    stations = get_stations(conn)
    conn.close()
    total_stations = len(stations)
    print(f"\n📍 {total_stations} stations to process\n")

    all_results = []

    for idx, srow in stations.iterrows():
        sid = srow['id']
        slat = srow['latitude']
        slon = srow['longitude']

        # Pre-filter: rough bounding box (±1.5 degrees ≈ ~150km)
        lat_mask = np.abs(fire_lats - slat) <= 1.5
        lon_mask = np.abs(fire_lons - slon) <= 1.5
        box_mask = lat_mask & lon_mask

        nearby_lats = fire_lats[box_mask]
        nearby_lons = fire_lons[box_mask]
        nearby_dates = fire_dates[box_mask]

        if len(nearby_lats) == 0:
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{total_stations}] {srow['name'][:30]:<30} → 0 fires nearby")
            continue

        # Exact haversine distance
        distances = haversine_vectorized(slat, slon, nearby_lats, nearby_lons)
        within_radius = distances <= RADIUS_KM

        # Count fires per day
        dates_in_radius = nearby_dates[within_radius]
        unique_dates, counts = np.unique(dates_in_radius, return_counts=True)

        for d, c in zip(unique_dates, counts):
            all_results.append({
                'station_id': sid,
                'date': d,
                'fire_count': int(c)
            })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{total_stations}] {srow['name'][:30]:<30} → {len(unique_dates)} days with fires")

    # Save
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ Saved {len(result_df):,} rows to {OUTPUT_CSV}")
    print(f"   Stations with fire data: {result_df['station_id'].nunique()}")
    print(f"   Date range: {result_df['date'].min()} → {result_df['date'].max()}")
    print(f"   Mean fire count: {result_df['fire_count'].mean():.1f}")
    print(f"   Max fire count:  {result_df['fire_count'].max()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
