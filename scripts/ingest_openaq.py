"""
IndiaAQ Intelligence — Scaled Data Ingestion
=============================================
Fetches ALL Indian AQ station data from OpenAQ v3 API
and writes directly to PostgreSQL.

Supports:
    - Auto-discover all India stations (no hardcoded list)
    - Paginated measurement fetching (1000 per page)
    - Date range filtering (2021-01-01 to present)
    - Checkpoint/resume (if script crashes, restart from last station)
    - Idempotent (safe to run twice — UNIQUE index prevents duplicates)

Usage:
    export OPENAQ_API_KEY="your_key_here"
    python scripts/ingest_openaq.py                      # full backfill
    python scripts/ingest_openaq.py --days 7              # last 7 days only
    python scripts/ingest_openaq.py --station-id 17       # single station
"""

import os
import sys
import json
import time
import argparse
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta, timezone

# ─── Config ───────────────────────────────────────────────
API_BASE = "https://api.openaq.org/v3"
INDIA_COUNTRY_ID = 9
DATE_FROM = "2021-01-01"
RATE_LIMIT_SLEEP = 0.25          # seconds between API calls
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".ingest_checkpoint.json")

# Database connection
DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": 5432,
}


def get_headers():
    """Get API headers with authentication."""
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise ValueError(
            "Set OPENAQ_API_KEY environment variable!\n"
            "  export OPENAQ_API_KEY='your_key_here'"
        )
    return {"X-API-Key": key}


def get_db_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


# ─── Step 1: Discover All India Stations ──────────────────
def fetch_india_stations(headers):
    """Fetch all active monitoring stations in India."""
    stations = []
    page = 1

    while True:
        print(f"  Fetching stations page {page}...")
        r = requests.get(
            f"{API_BASE}/locations",
            headers=headers,
            params={
                "countries_id": INDIA_COUNTRY_ID,
                "limit": 1000,
                "page": page,
            },
        )

        if r.status_code != 200:
            print(f"  ❌ API error {r.status_code}: {r.text[:200]}")
            break

        data = r.json()
        results = data.get("results", [])

        if not results:
            break

        for loc in results:
            stations.append({
                "openaq_id": loc["id"],
                "name": loc.get("name", f"Station-{loc['id']}"),
                "city": loc.get("locality") or loc.get("name", ""),
                "state": loc.get("country", {}).get("name", "India"),
                "latitude": loc.get("coordinates", {}).get("latitude"),
                "longitude": loc.get("coordinates", {}).get("longitude"),
            })

        # Check if more pages exist
        raw_found = data.get("meta", {}).get("found", 0)
        try:
            found = int(str(raw_found).replace(">", "").strip())
        except (ValueError, TypeError):
            found = 9999999  # assume more pages if unparseable
        if page * 1000 >= found:
            break

        page += 1
        time.sleep(RATE_LIMIT_SLEEP)

    return stations


def upsert_stations(conn, stations):
    """Insert stations into database (skip duplicates)."""
    sql = """
        INSERT INTO stations (openaq_id, name, city, state, latitude, longitude)
        VALUES %s
        ON CONFLICT (openaq_id) DO UPDATE SET
            name = EXCLUDED.name,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            updated_at = now()
    """
    values = [
        (s["openaq_id"], s["name"], s["city"], s["state"], s["latitude"], s["longitude"])
        for s in stations
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    print(f"  ✅ Upserted {len(stations)} stations")


def get_station_id_map(conn):
    """Get mapping of openaq_id → internal station id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, openaq_id FROM stations")
        return {row[1]: row[0] for row in cur.fetchall()}


# ─── Step 2: Fetch Measurements Per Station ───────────────
def fetch_station_sensors(station_openaq_id, headers):
    """Get all sensors for a station."""
    r = requests.get(
        f"{API_BASE}/locations/{station_openaq_id}/sensors",
        headers=headers,
    )
    if r.status_code != 200:
        return []
    return r.json().get("results", [])


def fetch_sensor_measurements(sensor_id, headers, date_from, date_to):
    """Fetch all measurements for a sensor with pagination."""
    all_measurements = []
    page = 1

    while True:
        params = {
            "limit": 1000,
            "page": page,
            "datetime_from": date_from,
            "datetime_to": date_to,
        }

        r = requests.get(
            f"{API_BASE}/sensors/{sensor_id}/measurements",
            headers=headers,
            params=params,
        )

        if r.status_code != 200:
            print(f"    ⚠️ Sensor {sensor_id} page {page}: HTTP {r.status_code}")
            break

        data = r.json()
        results = data.get("results", [])

        if not results:
            break

        all_measurements.extend(results)

        # Check if more pages
        raw_found = data.get("meta", {}).get("found", 0)
        try:
            found = int(str(raw_found).replace(">", "").strip())
        except (ValueError, TypeError):
            found = 9999999  # assume more pages if unparseable
        if page * 1000 >= found:
            break

        page += 1
        time.sleep(RATE_LIMIT_SLEEP)

    return all_measurements


def insert_measurements(conn, rows):
    """Bulk insert measurements into raw_measurements (skip duplicates)."""
    if not rows:
        return 0

    sql = """
        INSERT INTO raw_measurements
            (station_id, sensor_id, parameter, value, unit, datetime_utc, datetime_local)
        VALUES %s
        ON CONFLICT (station_id, parameter, datetime_utc) DO NOTHING
    """
    values = [
        (
            r["station_id"],
            r["sensor_id"],
            r["parameter"],
            r["value"],
            r["unit"],
            r["datetime_utc"],
            r["datetime_local"],
        )
        for r in rows
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    return len(values)


# ─── Step 3: Checkpoint (Resume After Crash) ──────────────
def load_checkpoint():
    """Load last successfully processed station."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"last_completed_openaq_id": None, "completed_count": 0}


def save_checkpoint(openaq_id, count):
    """Save progress."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_completed_openaq_id": openaq_id,
            "completed_count": count,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)


def clear_checkpoint():
    """Remove checkpoint file when done."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


# ─── Main Pipeline ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IndiaAQ Data Ingestion")
    parser.add_argument("--days", type=int, default=None,
                        help="Fetch only last N days (default: full backfill from 2021)")
    parser.add_argument("--station-id", type=int, default=None,
                        help="Fetch only this OpenAQ station ID")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    # Date range
    now = datetime.now(timezone.utc)
    date_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.days:
        date_from = (now - timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"📅 Fetching last {args.days} days")
    else:
        date_from = f"{DATE_FROM}T00:00:00Z"
        print(f"📅 Full backfill: {DATE_FROM} → today")

    headers = get_headers()
    conn = get_db_connection()

    # ── Step 1: Discover stations ──
    print("\n🔍 Step 1: Discovering India stations...")
    if args.station_id:
        stations = [{"openaq_id": args.station_id, "name": f"Station-{args.station_id}",
                      "city": "", "state": "India", "latitude": None, "longitude": None}]
    else:
        stations = fetch_india_stations(headers)

    print(f"  Found {len(stations)} stations in India")

    upsert_stations(conn, stations)
    id_map = get_station_id_map(conn)

    # ── Step 2: Fetch measurements ──
    print(f"\n📡 Step 2: Fetching measurements...")

    # Resume support
    checkpoint = load_checkpoint() if args.resume else {"last_completed_openaq_id": None, "completed_count": 0}
    skip_until = checkpoint["last_completed_openaq_id"]
    skipping = skip_until is not None
    completed = checkpoint["completed_count"]

    total_rows = 0
    total_stations = len(stations)

    for i, station in enumerate(stations):
        openaq_id = station["openaq_id"]

        # Skip already processed stations (resume mode)
        if skipping:
            if openaq_id == skip_until:
                skipping = False
            continue

        internal_id = id_map.get(openaq_id)
        if not internal_id:
            continue

        completed += 1
        print(f"\n  [{completed}/{total_stations}] {station['name']} (ID={openaq_id})")

        # Get sensors
        sensors = fetch_station_sensors(openaq_id, headers)
        if not sensors:
            print(f"    No sensors found")
            save_checkpoint(openaq_id, completed)
            continue

        station_rows = 0
        for sensor in sensors:
            sensor_id = sensor["id"]
            param = sensor["parameter"]["name"]
            unit = sensor["parameter"]["units"]

            measurements = fetch_sensor_measurements(sensor_id, headers, date_from, date_to)

            if not measurements:
                continue

            # Prepare rows
            rows = []
            for m in measurements:
                try:
                    rows.append({
                        "station_id": internal_id,
                        "sensor_id": sensor_id,
                        "parameter": param,
                        "value": m["value"],
                        "unit": unit,
                        "datetime_utc": m["period"]["datetimeFrom"]["utc"],
                        "datetime_local": m["period"]["datetimeFrom"].get("local", m["period"]["datetimeFrom"]["utc"]),
                    })
                except (KeyError, TypeError) as e:
                    continue  # skip malformed records

            inserted = insert_measurements(conn, rows)
            station_rows += inserted
            print(f"    {param}: {len(measurements)} fetched, {inserted} inserted")

            time.sleep(RATE_LIMIT_SLEEP)

        total_rows += station_rows
        print(f"    → Station total: {station_rows} rows")

        # Save checkpoint after each station
        save_checkpoint(openaq_id, completed)

    # ── Done ──
    print(f"\n{'='*50}")
    print(f"✅ Ingestion complete!")
    print(f"   Stations processed: {completed}")
    print(f"   Total rows inserted: {total_rows}")
    print(f"{'='*50}")

    # Verify
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM raw_measurements")
        total_db = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT station_id) FROM raw_measurements")
        stations_db = cur.fetchone()[0]
        cur.execute("SELECT MIN(datetime_utc), MAX(datetime_utc) FROM raw_measurements")
        date_range = cur.fetchone()

    print(f"\n📊 Database status:")
    print(f"   Total measurements: {total_db:,}")
    print(f"   Active stations: {stations_db}")
    print(f"   Date range: {date_range[0]} → {date_range[1]}")

    clear_checkpoint()
    conn.close()


if __name__ == "__main__":
    main()
