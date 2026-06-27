#!/usr/bin/env python3
"""
Partitioned AOD Backfill — Multi-VM Parallel Edition
=====================================================
Run this on multiple VMs (each with a different IP) to bypass
Open-Meteo's per-IP 10K/day rate limit.

Usage:
  python3 backfill_aod_partitioned.py --partition 0 --total 3
  python3 backfill_aod_partitioned.py --partition 1 --total 3
  python3 backfill_aod_partitioned.py --partition 2 --total 3

Each VM gets a non-overlapping slice of stations via:
  station_index % total_partitions == partition_id

Environment Variables (REQUIRED):
  AZURE_DB_HOST     = <your-db-host>.postgres.database.azure.com
  AZURE_DB_USER     = <your-db-user>
  AZURE_DB_PASSWORD = <your-db-password>
  AZURE_DB_NAME     = <your-db-name>

Self-contained: No project imports needed. Copy this single file to any VM.
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import date, timedelta

# ─── Attempt to import psycopg2 ───
try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    print("❌ psycopg2 not found. Install: pip3 install psycopg2-binary")
    sys.exit(1)

# ─── Constants ───
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
START_DATE = "2021-01-01"
END_DATE = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")


def get_db_config():
    """Read DB config from environment variables."""
    required = {
        "AZURE_DB_HOST": os.getenv("AZURE_DB_HOST"),
        "AZURE_DB_USER": os.getenv("AZURE_DB_USER"),
        "AZURE_DB_PASSWORD": os.getenv("AZURE_DB_PASSWORD"),
        "AZURE_DB_NAME": os.getenv("AZURE_DB_NAME", "indiaaq"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        print("Set them with: export AZURE_DB_HOST=... AZURE_DB_USER=... AZURE_DB_PASSWORD=... AZURE_DB_NAME=...")
        sys.exit(1)

    return {
        "dbname": required["AZURE_DB_NAME"],
        "user": required["AZURE_DB_USER"],
        "password": required["AZURE_DB_PASSWORD"],
        "host": required["AZURE_DB_HOST"],
        "port": 5432,
        "sslmode": "require",
        "connect_timeout": 30,
    }


def get_connection(db_config, label=""):
    """Create a new DB connection with retry logic."""
    for attempt in range(5):
        try:
            conn = psycopg2.connect(**db_config)
            return conn
        except psycopg2.OperationalError as e:
            wait = (attempt + 1) * 10
            print(f"⚠️ [{label}] DB connect failed (attempt {attempt+1}/5). Retry in {wait}s: {e}")
            time.sleep(wait)
    print(f"❌ [{label}] Could not connect to DB after 5 attempts. Exiting.")
    sys.exit(1)


def get_stations_with_missing_aod(conn):
    """Get all stations that still have NULL AOD values."""
    query = """
        SELECT DISTINCT s.id, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.date BETWEEN %s AND %s
          AND df.om_aerosol_optical_depth IS NULL
        ORDER BY s.id
    """
    return pd.read_sql(query, conn, params=(START_DATE, END_DATE))


def fetch_openmeteo_aod(lat, lon, station_id):
    """Fetch AOD from Open-Meteo Air Quality API, chunked by year."""
    all_dfs = []
    years = [2021, 2022, 2023, 2024, 2025, 2026]

    for year in years:
        start = f"{year}-01-01"
        if year == 2026:
            end = END_DATE
            if start > end:
                continue
        else:
            end = f"{year}-12-31"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ["aerosol_optical_depth"],
            "timezone": "auto",
        }

        try:
            resp = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=30)
        except requests.exceptions.Timeout:
            print(f"  ⏳ Timeout for station {station_id} year {year}, skipping chunk")
            time.sleep(5)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"  🔌 Connection error for station {station_id} year {year}: {e}")
            time.sleep(10)
            continue

        # Respect rate limits
        time.sleep(2.0)

        if resp.status_code == 429:
            raise Exception("Rate Limited (429)")
        if resp.status_code != 200:
            continue

        data = resp.json().get("hourly", {})
        aod_array = data.get("aerosol_optical_depth", [])
        time_array = data.get("time", [])

        if not aod_array or not time_array:
            continue

        df_hourly = pd.DataFrame({"time": pd.to_datetime(time_array), "aod": aod_array})
        df_hourly["date"] = df_hourly["time"].dt.date
        # Drop NaN before averaging
        df_hourly = df_hourly.dropna(subset=["aod"])
        if df_hourly.empty:
            continue

        df_daily = df_hourly.groupby("date")["aod"].mean().reset_index()
        df_daily.rename(columns={"aod": "om_aerosol_optical_depth"}, inplace=True)
        all_dfs.append(df_daily)

        time.sleep(0.2)

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def update_aod_bulk(conn, station_id, df):
    """Write AOD data to Azure DB. Only updates NULL cells (dedup safe)."""
    if df is None or df.empty:
        return 0

    sql = """
        UPDATE daily_features
        SET om_aerosol_optical_depth = %s
        WHERE station_id = %s AND date = %s
          AND om_aerosol_optical_depth IS NULL
    """

    df_valid = df.dropna(subset=["om_aerosol_optical_depth"])
    if df_valid.empty:
        return 0

    values = [(row["om_aerosol_optical_depth"], station_id, row["date"]) for _, row in df_valid.iterrows()]

    cur = conn.cursor()
    execute_batch(cur, sql, values, page_size=500)
    actual_updated = len(values)
    conn.commit()
    cur.close()
    return actual_updated


def process_station(db_config, station_row, index, my_total, label):
    """Process a single station with full error handling."""
    sid = int(station_row["id"])
    lat = float(station_row["latitude"])
    lon = float(station_row["longitude"])

    for attempt in range(5):
        conn = None
        try:
            conn = get_connection(db_config, label)
            df = fetch_openmeteo_aod(lat, lon, sid)
            updated = 0
            if df is not None:
                updated = update_aod_bulk(conn, sid, df)
            print(f"  [{index}/{my_total}] Station {sid} ✅ Updated {updated} rows.")
            return sid, updated, True

        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass

            err_str = str(e)
            if "429" in err_str:
                wait = 30 * (attempt + 1)
                print(f"  ⚠️ Rate limit on station {sid} (attempt {attempt+1}/5). Waiting {wait}s...")
                time.sleep(wait)
            elif "timeout" in err_str.lower():
                wait = 10 * (attempt + 1)
                print(f"  ⏳ Timeout on station {sid} (attempt {attempt+1}/5). Retry in {wait}s...")
                time.sleep(wait)
            elif "connection" in err_str.lower() or "operational" in err_str.lower():
                wait = 15 * (attempt + 1)
                print(f"  🔌 DB lost on station {sid} (attempt {attempt+1}/5). Reconnect in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Unknown error on station {sid}: {e}")
                break
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    return sid, 0, False


def main():
    parser = argparse.ArgumentParser(description="Partitioned AOD Backfill")
    parser.add_argument("--partition", type=int, required=True, help="This VM's partition ID (0-indexed)")
    parser.add_argument("--total", type=int, required=True, help="Total number of partitions (VMs)")
    args = parser.parse_args()

    pid = args.partition
    total_p = args.total
    label = f"VM-{pid}"

    print(f"{'='*60}")
    print(f"🚀 Partitioned AOD Backfill — {label} (Partition {pid}/{total_p})")
    print(f"📅 Date Range: {START_DATE} → {END_DATE}")
    print(f"{'='*60}")

    db_config = get_db_config()

    # Test connection first
    print(f"\n🔌 Testing Azure DB connection...")
    conn = get_connection(db_config, label)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_features WHERE om_aerosol_optical_depth IS NULL")
    total_null = cur.fetchone()[0]
    print(f"✅ Connected! Total NULL AOD rows remaining: {total_null:,}")

    # Get all stations with missing AOD
    all_stations = get_stations_with_missing_aod(conn)
    conn.close()

    total_stations = len(all_stations)
    print(f"🔍 Total stations with missing AOD: {total_stations}")

    # Partition: take every Nth station starting from our partition ID
    my_stations = all_stations.iloc[pid::total_p].reset_index(drop=True)
    my_count = len(my_stations)

    print(f"📦 {label} assigned {my_count} stations (out of {total_stations})")
    print(f"⚡ Starting processing...\n")

    total_updated = 0
    total_failed = 0
    start_time = time.time()

    for i, (_, row) in enumerate(my_stations.iterrows(), 1):
        sid, updated, success = process_station(db_config, row, i, my_count, label)
        if success:
            total_updated += updated
        else:
            total_failed += 1

        # Breathing room between stations
        time.sleep(1)

    elapsed = time.time() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"🎉 {label} Complete in {hrs}h {mins}m!")
    print(f"📈 Rows updated: {total_updated:,}")
    print(f"✅ Stations OK: {my_count - total_failed}/{my_count}")
    print(f"❌ Stations failed: {total_failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
