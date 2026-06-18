"""
UK DEFRA AURN — PM2.5 Data Ingest via SOS API
===============================================
Downloads hourly PM2.5 data from DEFRA's Sensor Observation Service,
aggregates to daily averages, and inserts into PostgreSQL.

Source: https://uk-air.defra.gov.uk/sos-ukair/api/v1/

Usage:
    python scripts/fetch_defra_bulk.py
"""

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

SOS_BASE = "https://uk-air.defra.gov.uk/sos-ukair/api/v1"
TIMESPAN = "2026-01-01T00:00:00Z/2026-06-17T23:59:59Z"


def get_pm25_timeseries():
    """Get all PM2.5 timeseries from DEFRA SOS API."""
    print("  Fetching PM2.5 timeseries metadata...")
    r = requests.get(f"{SOS_BASE}/timeseries", 
        params={"expanded": "true"}, timeout=60)
    r.raise_for_status()
    all_ts = r.json()
    
    # Filter PM2.5 (pollutant 6001 = PM2.5)
    pm25_ts = [t for t in all_ts if "2.5" in t.get("label", "")]
    print(f"  Found {len(pm25_ts)} PM2.5 timeseries out of {len(all_ts)} total")
    return pm25_ts


def fetch_timeseries_data(ts_id, timespan):
    """Fetch hourly data for one timeseries."""
    r = requests.get(f"{SOS_BASE}/timeseries/{ts_id}/getData",
        params={"timespan": timespan}, timeout=60)
    r.raise_for_status()
    return r.json().get("values", [])


def extract_station_info(ts):
    """Extract station name and coordinates from timeseries metadata."""
    station = ts.get("station", {})
    props = station.get("properties", {})
    geom = station.get("geometry", {})
    coords = geom.get("coordinates", [None, None])
    
    label = props.get("label", f"DEFRA-{ts['id']}")
    # Clean label: "Aberdeen-Particulate matter less than 2.5..." -> "Aberdeen"
    name = label.split("-Particulate")[0].split(" - ")[0].strip()
    
    return {
        "name": f"DEFRA-{name}",
        "lat": coords[0] if len(coords) > 0 else None,
        "lon": coords[1] if len(coords) > 1 else None,
        "label": label,
    }


def upsert_stations(conn, stations_info):
    """Insert DEFRA stations and return name->id mapping."""
    with conn.cursor() as cur:
        inserted = 0
        for info in stations_info:
            if info["lat"] is None or info["lon"] is None:
                continue
            cur.execute("""
                INSERT INTO stations (name, city, country_code, latitude, longitude, is_active)
                VALUES (%s, %s, 'GB', %s, %s, true)
                ON CONFLICT (name, country_code) DO NOTHING
                RETURNING id
            """, (
                info["name"],
                "",
                float(info["lat"]),
                float(info["lon"]),
            ))
            if cur.fetchone():
                inserted += 1
        conn.commit()
        
        # Get full mapping
        cur.execute("SELECT name, id FROM stations WHERE country_code = 'GB' AND name LIKE 'DEFRA-%'")
        mapping = dict(cur.fetchall())
    
    print(f"  Stations: {inserted} new, {len(mapping)} total DEFRA stations")
    return mapping


def insert_daily_measurements(conn, daily_records, station_mapping):
    """Insert daily-aggregated measurements."""
    values = []
    for record in daily_records:
        station_id = station_mapping.get(record["station_name"])
        if station_id is None:
            continue
        dt_str = f"{record['date']}T12:00:00+00:00"
        values.append((
            station_id, 0, "pm25",
            float(record["pm25"]),
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
        execute_values(cur, sql, values)
    conn.commit()
    return len(values)


def main():
    start = time.time()
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Count before
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id WHERE s.country_code = 'GB'
        """)
        before = cur.fetchone()[0]
        print(f"\nGB measurements before: {before:,}")
    
    # Phase 1: Get PM2.5 timeseries
    print(f"\n{'='*60}")
    print("Phase 1: Discover DEFRA PM2.5 stations")
    print(f"{'='*60}")
    pm25_ts = get_pm25_timeseries()
    
    # Extract station info
    stations_info = [extract_station_info(ts) for ts in pm25_ts]
    
    # Phase 2: Upsert stations
    print(f"\n{'='*60}")
    print("Phase 2: Upsert Stations")
    print(f"{'='*60}")
    station_mapping = upsert_stations(conn, stations_info)
    
    # Phase 3: Fetch data for each timeseries
    print(f"\n{'='*60}")
    print(f"Phase 3: Fetch hourly data ({TIMESPAN[:10]} to {TIMESPAN[25:35]})")
    print(f"{'='*60}")
    
    all_daily = []
    for i, ts in enumerate(pm25_ts):
        ts_id = ts["id"]
        info = extract_station_info(ts)
        
        try:
            values = fetch_timeseries_data(ts_id, TIMESPAN)
        except Exception as e:
            print(f"  [{i+1}/{len(pm25_ts)}] {info['name']}: ERROR {e}")
            continue
        
        if not values:
            continue
        
        # Convert hourly to daily averages
        hourly = pd.DataFrame(values)
        hourly["datetime"] = pd.to_datetime(hourly["timestamp"], unit="ms")
        hourly["date"] = hourly["datetime"].dt.date
        
        daily = hourly.groupby("date")["value"].mean().reset_index()
        daily = daily[daily["value"] > 0]  # Filter negative/zero readings
        
        for _, row in daily.iterrows():
            all_daily.append({
                "station_name": info["name"],
                "date": str(row["date"]),
                "pm25": row["value"],
            })
        
        if (i + 1) % 20 == 0 or len(values) > 0:
            print(f"  [{i+1}/{len(pm25_ts)}] {info['name']}: {len(values)} hourly → {len(daily)} daily")
    
    print(f"\n  Total daily records: {len(all_daily):,}")
    
    # Phase 4: Insert measurements
    print(f"\n{'='*60}")
    print("Phase 4: Insert Measurements")
    print(f"{'='*60}")
    inserted = insert_daily_measurements(conn, all_daily, station_mapping)
    
    # Final stats
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id WHERE s.country_code = 'GB'
        """)
        after = cur.fetchone()[0]
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print("DEFRA INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Before: {before:,} GB measurements")
    print(f"  After:  {after:,} GB measurements")
    print(f"  Added:  {after - before:,} new rows")
    print(f"  Time:   {int(elapsed // 60)}m {int(elapsed % 60)}s")
    
    conn.close()


if __name__ == "__main__":
    main()
