"""
Australia NSW EPA — PM2.5 Data Ingest
======================================
Downloads daily PM2.5 averages from NSW Air Quality API,
inserts into PostgreSQL.

Source: https://data.airquality.nsw.gov.au/api/Data/
API: POST-based, swagger at /swagger/v1/swagger.json

Usage:
    python scripts/fetch_nsw_bulk.py
    python scripts/fetch_nsw_bulk.py --start-year 2023
"""

import os
import sys
import time
import argparse
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

NSW_API = "https://data.airquality.nsw.gov.au/api/Data"


def get_nsw_sites():
    """Get all NSW monitoring sites."""
    r = requests.get(f"{NSW_API}/get_SiteDetails", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_daily_pm25(site_ids, start_date, end_date):
    """
    Fetch daily PM2.5 averages for given sites and date range.
    API accepts POST with JSON body.
    """
    payload = {
        "Parameters": ["PM2.5"],
        "Sites": site_ids,
        "StartDate": start_date,
        "EndDate": end_date,
        "Categories": ["Averages"],
        "SubCategories": ["Daily"],
        "Frequency": ["24h average derived from 1h average"],
    }

    r = requests.post(f"{NSW_API}/get_Observations", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def upsert_stations(conn, sites):
    """Insert NSW stations into DB."""
    with conn.cursor() as cur:
        inserted = 0
        for site in sites:
            if site.get("Latitude") is None or site.get("Longitude") is None:
                continue
            name = f"NSW-{site['SiteName']}"
            cur.execute("""
                INSERT INTO stations (name, city, country_code, latitude, longitude, is_active)
                VALUES (%s, %s, 'AU', %s, %s, true)
                ON CONFLICT (name, country_code) DO NOTHING
                RETURNING id
            """, (
                name,
                site.get("Region", ""),
                float(site["Latitude"]),
                float(site["Longitude"]),
            ))
            if cur.fetchone():
                inserted += 1
        conn.commit()

        cur.execute("SELECT name, id FROM stations WHERE country_code = 'AU' AND name LIKE 'NSW-%'")
        mapping = dict(cur.fetchall())

    print(f"  Stations: {inserted} new, {len(mapping)} total NSW stations")
    return mapping


def insert_measurements(conn, records, station_mapping):
    """Insert daily PM2.5 measurements."""
    values = []
    for rec in records:
        if rec.get("Value") is None:
            continue
        site_name = station_mapping.get(rec["site_name"])
        if site_name is None:
            continue

        dt_str = f"{rec['Date']}T12:00:00+10:00"
        values.append((
            site_name, 0, "pm25",
            float(rec["Value"]),
            "µg/m³", dt_str, dt_str,
        ))

    if not values:
        return 0

    sql = """
        INSERT INTO raw_measurements
            (station_id, sensor_id, parameter, value, unit, datetime_utc, datetime_local)
        VALUES %s
        ON CONFLICT (station_id, parameter, datetime_utc) DO NOTHING
    """
    with conn.cursor() as cur:
        batch_size = 5000
        for i in range(0, len(values), batch_size):
            execute_values(cur, sql, values[i:i + batch_size])
        conn.commit()

    return len(values)


def main():
    parser = argparse.ArgumentParser(description="Fetch NSW PM2.5 data")
    parser.add_argument("--start-year", type=int, default=2021,
                        help="Start year (default: 2021)")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="End year (default: 2025)")
    args = parser.parse_args()

    start = time.time()
    conn = psycopg2.connect(**DB_CONFIG)

    # Count before
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id WHERE s.country_code = 'AU'
        """)
        before = cur.fetchone()[0]
        print(f"\nAU measurements before: {before:,}")

    # Phase 1: Get sites
    print(f"\n{'='*60}")
    print("Phase 1: Discover NSW monitoring sites")
    print(f"{'='*60}")
    sites = get_nsw_sites()
    print(f"  Found {len(sites)} NSW monitoring sites")
    site_ids = [s["Site_Id"] for s in sites]

    # Build site_id -> name mapping
    site_name_map = {s["Site_Id"]: f"NSW-{s['SiteName']}" for s in sites}

    # Phase 2: Upsert stations
    print(f"\n{'='*60}")
    print("Phase 2: Upsert Stations")
    print(f"{'='*60}")
    station_mapping = upsert_stations(conn, sites)

    # Phase 3: Fetch data year by year (API might have limits on date ranges)
    print(f"\n{'='*60}")
    print(f"Phase 3: Fetch PM2.5 data ({args.start_year}-{args.end_year})")
    print(f"{'='*60}")

    total_inserted = 0

    for year in range(args.start_year, args.end_year + 1):
        # Fetch 6 months at a time to stay within API limits
        for half in range(2):
            if half == 0:
                start_date = f"{year}-01-01"
                end_date = f"{year}-06-30"
            else:
                start_date = f"{year}-07-01"
                end_date = f"{year}-12-31"

            # Don't fetch future dates
            if start_date > datetime.now().strftime("%Y-%m-%d"):
                continue

            print(f"  Fetching {start_date} → {end_date}...", end=" ", flush=True)

            try:
                data = fetch_daily_pm25(site_ids, start_date, end_date)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # Filter valid values and attach station names
            valid = []
            for obs in data:
                if obs.get("Value") is not None and obs["Value"] > 0:
                    obs["site_name"] = site_name_map.get(obs["Site_Id"])
                    if obs["site_name"]:
                        valid.append(obs)

            # Insert
            inserted = insert_measurements(conn, valid, station_mapping)
            total_inserted += inserted
            print(f"{len(data)} obs → {len(valid)} valid → {inserted} inserted")

        time.sleep(1)  # Be nice to the API

    # Final stats
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id WHERE s.country_code = 'AU'
        """)
        after = cur.fetchone()[0]

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print("NSW INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Before: {before:,} AU measurements")
    print(f"  After:  {after:,} AU measurements")
    print(f"  Added:  {after - before:,} new rows")
    print(f"  Time:   {int(elapsed // 60)}m {int(elapsed % 60)}s")

    conn.close()


if __name__ == "__main__":
    main()
