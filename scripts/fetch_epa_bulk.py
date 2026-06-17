"""
US EPA AQS — Bulk PM2.5 Data Ingest
====================================
Downloads pre-generated annual CSV files from EPA AQS,
maps to our schema, and inserts into PostgreSQL.

Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html
File pattern: daily_88101_{YEAR}.zip

Usage:
    python scripts/fetch_epa_bulk.py
    python scripts/fetch_epa_bulk.py --years 2024 2025
"""

import os
import sys
import zipfile
import argparse
import io
import time
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests

DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": 5432,
}

EPA_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "epa")
DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]


def download_epa_file(year):
    """Download annual EPA PM2.5 zip file if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"daily_88101_{year}.zip")
    csv_path = os.path.join(DATA_DIR, f"daily_88101_{year}.csv")

    if os.path.exists(csv_path):
        print(f"  ✓ {year} CSV already exists, skipping download")
        return csv_path

    if not os.path.exists(zip_path):
        url = f"{EPA_BASE_URL}/daily_88101_{year}.zip"
        print(f"  Downloading {url}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB")

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print(f"  ✓ Extracted {year}")

    return csv_path


def parse_epa_csv(csv_path, year):
    """
    Parse EPA CSV and return standardized DataFrame.

    We only keep '24 HOUR' and '24-HR BLK AVG' sample durations
    (daily averages), skip hourly data to avoid duplicates.

    Station ID format: EPA-{state_code}-{county_code}-{site_num}
    """
    print(f"  Parsing {year} CSV...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to daily averages only (skip hourly to avoid duplication)
    daily_mask = df["Sample Duration"].isin(["24 HOUR", "24-HR BLK AVG"])
    df = df[daily_mask].copy()

    # If both '24 HOUR' and '24-HR BLK AVG' exist for same site+date, keep one
    # Prefer '24 HOUR' (filter method), group by site+date and take mean
    df["site_key"] = (
        df["State Code"].astype(str).str.zfill(2) + "-" +
        df["County Code"].astype(str).str.zfill(3) + "-" +
        df["Site Num"].astype(str).str.zfill(4)
    )

    # Aggregate: one row per site per date
    agg = df.groupby(["site_key", "Date Local"]).agg({
        "Arithmetic Mean": "mean",
        "Latitude": "first",
        "Longitude": "first",
        "Local Site Name": "first",
        "State Name": "first",
        "County Name": "first",
        "City Name": "first",
    }).reset_index()

    agg.rename(columns={
        "Date Local": "date",
        "Arithmetic Mean": "pm25",
    }, inplace=True)

    print(f"  {year}: {len(agg):,} daily records from {agg['site_key'].nunique()} sites")
    return agg


def upsert_stations(conn, sites_df):
    """
    Insert/update EPA stations into stations table.
    Returns mapping: site_key -> station_id (internal DB ID).
    """
    # Get unique sites
    sites = sites_df.groupby("site_key").agg({
        "Latitude": "first",
        "Longitude": "first",
        "Local Site Name": "first",
        "State Name": "first",
        "City Name": "first",
    }).reset_index()

    with conn.cursor() as cur:
        # Get existing EPA stations
        cur.execute("""
            SELECT name, id FROM stations
            WHERE country_code = 'US' AND name LIKE 'EPA-%'
        """)
        existing = dict(cur.fetchall())

        inserted = 0
        for _, row in sites.iterrows():
            station_name = f"EPA-{row['site_key']}"
            if station_name in existing:
                continue

            display_name = row.get("Local Site Name", station_name)
            if pd.isna(display_name):
                display_name = station_name

            city = row.get("City Name", "")
            state = row.get("State Name", "")
            if pd.isna(city):
                city = ""
            if pd.isna(state):
                state = ""

            cur.execute("""
                INSERT INTO stations (name, city, country_code, latitude, longitude, is_active)
                VALUES (%s, %s, 'US', %s, %s, true)
                ON CONFLICT (name, country_code) DO NOTHING
                RETURNING id
            """, (
                station_name,
                f"{city}, {state}".strip(", "),
                float(row["Latitude"]),
                float(row["Longitude"]),
            ))
            result = cur.fetchone()
            if result:
                existing[station_name] = result[0]
                inserted += 1

        conn.commit()

        # Rebuild full mapping
        cur.execute("""
            SELECT name, id FROM stations
            WHERE country_code = 'US' AND name LIKE 'EPA-%'
        """)
        mapping = dict(cur.fetchall())

    print(f"  Stations: {inserted} new, {len(mapping)} total EPA stations")
    return mapping


def insert_measurements(conn, records_df, station_mapping):
    """Insert EPA measurements into raw_measurements table."""
    values = []
    skipped = 0

    for _, row in records_df.iterrows():
        station_name = f"EPA-{row['site_key']}"
        station_id = station_mapping.get(station_name)
        if station_id is None:
            skipped += 1
            continue

        dt_str = f"{row['date']}T12:00:00+00:00"
        values.append((
            station_id,     # station_id
            0,              # sensor_id (placeholder)
            "pm25",         # parameter
            float(row["pm25"]),  # value
            "µg/m³",        # unit
            dt_str,         # datetime_utc
            dt_str,         # datetime_local
        ))

    if not values:
        print(f"  No values to insert (skipped={skipped})")
        return 0

    sql = """
        INSERT INTO raw_measurements
            (station_id, sensor_id, parameter, value, unit, datetime_utc, datetime_local)
        VALUES %s
        ON CONFLICT (station_id, parameter, datetime_utc) DO NOTHING
    """

    with conn.cursor() as cur:
        batch_size = 10000
        total_inserted = 0
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            execute_values(cur, sql, batch)
            total_inserted += len(batch)
            if (i + batch_size) % 50000 == 0:
                conn.commit()
                print(f"    Progress: {total_inserted:,}/{len(values):,}")
        conn.commit()

    print(f"  Inserted {total_inserted:,} measurements (skipped {skipped:,})")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Fetch US EPA PM2.5 bulk data")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS,
                        help="Years to fetch (default: 2021-2025)")
    args = parser.parse_args()

    start = time.time()
    conn = psycopg2.connect(**DB_CONFIG)

    # Show starting state
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id
            WHERE s.country_code = 'US'
        """)
        before_count = cur.fetchone()[0]
        print(f"\nUS measurements before: {before_count:,}")

    all_records = []

    # Phase 1: Download and parse all years
    print(f"\n{'='*60}")
    print(f"Phase 1: Download EPA PM2.5 data ({args.years[0]}-{args.years[-1]})")
    print(f"{'='*60}")
    for year in args.years:
        try:
            csv_path = download_epa_file(year)
            records = parse_epa_csv(csv_path, year)
            all_records.append(records)
        except Exception as e:
            print(f"  ⚠️ {year} failed: {e}")

    if not all_records:
        print("No data downloaded!")
        return

    combined = pd.concat(all_records, ignore_index=True)
    print(f"\nTotal: {len(combined):,} daily records, {combined['site_key'].nunique()} sites")
    print(f"Date range: {combined['date'].min()} → {combined['date'].max()}")

    # Phase 2: Upsert stations
    print(f"\n{'='*60}")
    print("Phase 2: Upsert Stations")
    print(f"{'='*60}")
    station_mapping = upsert_stations(conn, combined)

    # Phase 3: Insert measurements
    print(f"\n{'='*60}")
    print("Phase 3: Insert Measurements")
    print(f"{'='*60}")
    total = insert_measurements(conn, combined, station_mapping)

    # Final stats
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id
            WHERE s.country_code = 'US'
        """)
        after_count = cur.fetchone()[0]

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"EPA INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Before: {before_count:,} US measurements")
    print(f"  After:  {after_count:,} US measurements")
    print(f"  Added:  {after_count - before_count:,} new rows")
    print(f"  Time:   {int(elapsed // 60)}m {int(elapsed % 60)}s")

    conn.close()


if __name__ == "__main__":
    main()
