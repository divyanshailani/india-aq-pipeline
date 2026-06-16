"""
Multi-Country AQ Data Fetcher (OpenAQ v3)
==========================================
Generic version of fetch_openaq_india.py that accepts any country.

Supports:
    - Auto-discover all stations for a country
    - Paginated measurement fetching (1000 per page)
    - Date range filtering
    - Checkpoint/resume (safe to restart)
    - Idempotent (UNIQUE index prevents duplicates)

Usage:
    export OPENAQ_API_KEY="your_key_here"
    python scripts/fetch_openaq.py --country US --days 7
    python scripts/fetch_openaq.py --country GB --days 30
    python scripts/fetch_openaq.py --country CN              # full backfill
    python scripts/fetch_openaq.py --country IN --days 7     # India incremental
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

# ── Country Config ────────────────────────────────────────
COUNTRIES = {
    "IN": {"openaq_id": 9,    "name": "India"},
    "US": {"openaq_id": 155,  "name": "United States"},
    "GB": {"openaq_id": 79,   "name": "United Kingdom"},
    "AU": {"openaq_id": 177,  "name": "Australia"},
}

# ── Config ────────────────────────────────────────────────
API_BASE = "https://api.openaq.org/v3"
DATE_FROM = "2021-01-01"
RATE_LIMIT_SLEEP = 0.25
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), ".checkpoints")

DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": 5432,
}


def get_headers():
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise ValueError("Set OPENAQ_API_KEY environment variable!")
    return {"X-API-Key": key}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_checkpoint_file(country_code):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{country_code}.json")


# ── Step 1: Discover Stations ─────────────────────────────
def fetch_country_stations(country_code, headers):
    """Fetch all monitoring stations for a country."""
    country_info = COUNTRIES[country_code]
    stations = []
    page = 1

    while True:
        print(f"  Fetching stations page {page}...")
        r = requests.get(
            f"{API_BASE}/locations",
            headers=headers,
            params={
                "countries_id": country_info["openaq_id"],
                "limit": 1000,
                "page": page,
            },
        )

        if r.status_code != 200:
            print(f"  API error {r.status_code}: {r.text[:200]}")
            break

        data = r.json()
        results = data.get("results", [])

        if not results:
            break

        for loc in results:
            stations.append({
                "openaq_id": loc["id"],
                "name": loc.get("name") or f"Station-{loc['id']}",
                "city": loc.get("locality") or loc.get("name") or f"Unknown-{loc['id']}",
                "state": (loc.get("country") or {}).get("name") or country_info["name"],
                "country_code": country_code,
                "latitude": loc.get("coordinates", {}).get("latitude"),
                "longitude": loc.get("coordinates", {}).get("longitude"),
            })

        raw_found = data.get("meta", {}).get("found", 0)
        try:
            found = int(str(raw_found).replace(">", "").strip())
        except (ValueError, TypeError):
            found = 9999999
        if page * 1000 >= found:
            break

        page += 1
        time.sleep(RATE_LIMIT_SLEEP)

    return stations


def upsert_stations(conn, stations):
    """Insert/update stations in database."""
    if not stations:
        return

    sql = """
        INSERT INTO stations (openaq_id, name, city, state, country_code, latitude, longitude)
        VALUES %s
        ON CONFLICT (openaq_id) DO UPDATE SET
            name = EXCLUDED.name,
            country_code = EXCLUDED.country_code,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            updated_at = now()
    """
    values = [
        (s["openaq_id"], s["name"], s["city"], s["state"],
         s["country_code"], s["latitude"], s["longitude"])
        for s in stations
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    print(f"  Upserted {len(stations)} stations")


def get_station_id_map(conn, country_code):
    """Get openaq_id -> internal id mapping for a country."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, openaq_id FROM stations WHERE country_code = %s",
            (country_code,)
        )
        return {row[1]: row[0] for row in cur.fetchall()}


# ── Step 2: Fetch Measurements ────────────────────────────
def fetch_station_sensors(station_openaq_id, headers):
    r = requests.get(
        f"{API_BASE}/locations/{station_openaq_id}/sensors",
        headers=headers,
    )
    if r.status_code != 200:
        return []
    return r.json().get("results", [])


def fetch_sensor_measurements(sensor_id, headers, date_from, date_to):
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
            break

        data = r.json()
        results = data.get("results", [])

        if not results:
            break

        all_measurements.extend(results)

        raw_found = data.get("meta", {}).get("found", 0)
        try:
            found = int(str(raw_found).replace(">", "").strip())
        except (ValueError, TypeError):
            found = 9999999
        if page * 1000 >= found:
            break

        page += 1
        time.sleep(RATE_LIMIT_SLEEP)

    return all_measurements


def insert_measurements(conn, rows):
    if not rows:
        return 0

    sql = """
        INSERT INTO raw_measurements
            (station_id, sensor_id, parameter, value, unit, datetime_utc, datetime_local)
        VALUES %s
        ON CONFLICT (station_id, parameter, datetime_utc) DO NOTHING
    """
    values = [
        (r["station_id"], r["sensor_id"], r["parameter"],
         r["value"], r["unit"], r["datetime_utc"], r["datetime_local"])
        for r in rows
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    return len(values)


# ── Checkpoint ────────────────────────────────────────────
def load_checkpoint(country_code):
    path = get_checkpoint_file(country_code)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"last_completed_openaq_id": None, "completed_count": 0}


def save_checkpoint(country_code, openaq_id, count):
    path = get_checkpoint_file(country_code)
    with open(path, "w") as f:
        json.dump({
            "last_completed_openaq_id": openaq_id,
            "completed_count": count,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)


def clear_checkpoint(country_code):
    path = get_checkpoint_file(country_code)
    if os.path.exists(path):
        os.remove(path)


# ── Main ──────────────────────────────────────────────────
def run_fetch(country_code, days=None, date_from=None, date_to=None, resume=False):
    """
    Main fetch function. Can be called from CLI or from orchestrator.

    Args:
        country_code: 'US', 'GB', 'CN', 'IN'
        days: fetch last N days (None = full backfill from 2021)
        date_from: explicit start date string 'YYYY-MM-DD' (overrides days)
        date_to: explicit end date string 'YYYY-MM-DD' (overrides days)
        resume: continue from checkpoint

    Returns:
        dict with stats
    """
    if country_code not in COUNTRIES:
        raise ValueError(f"Unknown country: {country_code}. Available: {list(COUNTRIES.keys())}")

    country_name = COUNTRIES[country_code]["name"]

    now = datetime.now(timezone.utc)

    # Priority: explicit dates > days > full backfill
    if date_from and date_to:
        dt_from = f"{date_from}T00:00:00Z"
        dt_to = f"{date_to}T23:59:59Z"
    elif days:
        dt_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_from = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        dt_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_from = f"{DATE_FROM}T00:00:00Z"

    print(f"\n{'='*60}")
    print(f"  {country_name} ({country_code}) -- AQ Data Fetch")
    print(f"  Date range: {dt_from[:10]} to {dt_to[:10]}")
    print(f"{'='*60}")

    headers = get_headers()
    conn = get_db_connection()

    # Step 1: Discover stations
    print(f"\n  Discovering {country_name} stations...")
    stations = fetch_country_stations(country_code, headers)
    print(f"  Found {len(stations)} stations")

    upsert_stations(conn, stations)
    id_map = get_station_id_map(conn, country_code)

    # Step 2: Fetch measurements
    print(f"\n  Fetching measurements...")

    checkpoint = load_checkpoint(country_code) if resume else {
        "last_completed_openaq_id": None, "completed_count": 0
    }
    skip_until = checkpoint["last_completed_openaq_id"]
    skipping = skip_until is not None
    completed = checkpoint["completed_count"]

    total_rows = 0
    total_stations = len(stations)

    for i, station in enumerate(stations):
        openaq_id = station["openaq_id"]

        if skipping:
            if openaq_id == skip_until:
                skipping = False
            continue

        internal_id = id_map.get(openaq_id)
        if not internal_id:
            continue

        completed += 1
        print(f"\n  [{completed}/{total_stations}] {station['name'][:50]} (ID={openaq_id})")

        sensors = fetch_station_sensors(openaq_id, headers)
        if not sensors:
            save_checkpoint(country_code, openaq_id, completed)
            continue

        station_rows = 0
        for sensor in sensors:
            sensor_id = sensor["id"]
            param = sensor["parameter"]["name"]
            unit = sensor["parameter"]["units"]

            measurements = fetch_sensor_measurements(sensor_id, headers, dt_from, dt_to)

            if not measurements:
                continue

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
                        "datetime_local": m["period"]["datetimeFrom"].get(
                            "local", m["period"]["datetimeFrom"]["utc"]
                        ),
                    })
                except (KeyError, TypeError):
                    continue

            inserted = insert_measurements(conn, rows)
            station_rows += inserted
            print(f"    {param}: {len(measurements)} fetched, {inserted} inserted")
            time.sleep(RATE_LIMIT_SLEEP)

        total_rows += station_rows
        save_checkpoint(country_code, openaq_id, completed)

    # Summary
    stats = {
        "country": country_code,
        "stations_found": len(stations),
        "stations_processed": completed,
        "rows_inserted": total_rows,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  {country_name} complete: {completed} stations, {total_rows} rows inserted")
    clear_checkpoint(country_code)
    conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Multi-Country AQ Data Fetcher")
    parser.add_argument("--country", type=str, required=True,
                        choices=list(COUNTRIES.keys()),
                        help="Country code: IN, US, GB, CN")
    parser.add_argument("--days", type=int, default=None,
                        help="Fetch last N days (default: full backfill)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    stats = run_fetch(args.country, args.days, args.resume)

    # Verify
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.country_code, COUNT(DISTINCT s.id) as stations,
                   COUNT(r.id) as measurements
            FROM stations s
            LEFT JOIN raw_measurements r ON s.id = r.station_id
            GROUP BY s.country_code
            ORDER BY s.country_code
        """)
        print(f"\n  Database status:")
        for row in cur.fetchall():
            print(f"    {row[0]}: {row[1]} stations, {row[2]:,} measurements")
    conn.close()


if __name__ == "__main__":
    main()
