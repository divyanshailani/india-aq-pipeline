"""
Process FIRMS fire data for GB and US.
Same logic as process_firms_fire.py but for global stations.
"""

import numpy as np
import pandas as pd
import psycopg2

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

RADIUS_KM = 100

FIRE_FILES = {
    "GB": {
        "archive": "data/raw/DL_FIRE_SV-C2_762578/fire_archive_SV-C2_762578.csv",
        "nrt": "data/raw/DL_FIRE_SV-C2_762578/fire_nrt_SV-C2_762578.csv",
    },
    "US": {
        "archive": "data/raw/DL_FIRE_SV-C2_762579/fire_archive_SV-C2_762579.csv",
        "nrt": "data/raw/DL_FIRE_SV-C2_762579/fire_nrt_SV-C2_762579.csv",
    },
    # AU will be added when user downloads it
}


def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2_arr)
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_stations_for_country(conn, country_code):
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.parameter = 'pm25'
          AND s.country_code = %s
          AND s.latitude IS NOT NULL
        ORDER BY s.id
    """, (country_code,))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["id", "name", "latitude", "longitude"])


def process_country(country_code, archive_path, nrt_path, conn):
    print(f"\n{'=' * 60}")
    print(f"  Processing FIRMS for {country_code}")
    print(f"{'=' * 60}")

    # Load fire data
    fires_archive = pd.read_csv(archive_path, usecols=['latitude', 'longitude', 'acq_date', 'confidence'])
    fires_nrt = pd.read_csv(nrt_path, usecols=['latitude', 'longitude', 'acq_date', 'confidence'])
    fires = pd.concat([fires_archive, fires_nrt], ignore_index=True)
    fires['acq_date'] = pd.to_datetime(fires['acq_date']).dt.date
    fires = fires[fires['confidence'].isin(['n', 'h'])]
    print(f"  Fire points: {len(fires):,}")
    print(f"  Date range: {fires['acq_date'].min()} → {fires['acq_date'].max()}")

    fire_lats = fires['latitude'].values
    fire_lons = fires['longitude'].values
    fire_dates = fires['acq_date'].values

    stations = get_stations_for_country(conn, country_code)
    total = len(stations)
    print(f"  Stations: {total}")

    all_results = []
    for idx, srow in stations.iterrows():
        sid, slat, slon = srow['id'], srow['latitude'], srow['longitude']

        lat_mask = np.abs(fire_lats - slat) <= 1.5
        lon_mask = np.abs(fire_lons - slon) <= 1.5
        box_mask = lat_mask & lon_mask

        nearby_lats = fire_lats[box_mask]
        nearby_lons = fire_lons[box_mask]
        nearby_dates = fire_dates[box_mask]

        if len(nearby_lats) == 0:
            continue

        distances = haversine_vectorized(slat, slon, nearby_lats, nearby_lons)
        within = distances <= RADIUS_KM
        dates_in = nearby_dates[within]
        unique_dates, counts = np.unique(dates_in, return_counts=True)

        for d, c in zip(unique_dates, counts):
            all_results.append({'station_id': sid, 'date': d, 'fire_count': int(c)})

        if (idx + 1) % 20 == 0:
            print(f"    [{idx+1}/{total}] {srow['name'][:30]:<30} → {len(unique_dates)} days")

    result_df = pd.DataFrame(all_results)
    output = f"data/fire_counts_{country_code.lower()}.csv"
    result_df.to_csv(output, index=False)
    print(f"  ✅ Saved {len(result_df):,} rows to {output}")
    return result_df


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    for cc, files in FIRE_FILES.items():
        process_country(cc, files["archive"], files["nrt"], conn)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
