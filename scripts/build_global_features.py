"""
Global AQ — Full ETL + Feature Engineering Pipeline
=====================================================
Runs cleaning + features for ALL countries, then merges:
  - NASA POWER weather data
  - NASA FIRMS fire counts
  - Adds country_code to daily_features

Usage: python scripts/build_global_features.py
"""

import sys
import os
import time
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cleaning import run_cleaning_pipeline
from src.features import run_feature_pipeline

DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": 5432,
}

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")


def add_country_column(conn):
    """Add country_code column to daily_features if missing."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'daily_features' AND column_name = 'country_code'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE daily_features ADD COLUMN country_code TEXT")
            conn.commit()
            print("  Added country_code column to daily_features")

        # Backfill country_code from stations table
        cur.execute("""
            UPDATE daily_features df
            SET country_code = s.country_code
            FROM stations s
            WHERE df.station_id = s.id
              AND df.country_code IS NULL
        """)
        updated = cur.rowcount
        conn.commit()
        if updated > 0:
            print(f"  Backfilled country_code for {updated:,} rows")


def add_nasa_columns(conn):
    """Add NASA weather + fire columns to daily_features if missing."""
    new_cols = {
        "nasa_temperature": "DOUBLE PRECISION",
        "nasa_humidity": "DOUBLE PRECISION",
        "nasa_wind_speed": "DOUBLE PRECISION",
        "precipitation": "DOUBLE PRECISION",
        "wind_direction": "DOUBLE PRECISION",
        "fire_count": "INTEGER",
    }
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'daily_features'
        """)
        existing = {row[0] for row in cur.fetchall()}

        for col, dtype in new_cols.items():
            if col not in existing:
                cur.execute(f"ALTER TABLE daily_features ADD COLUMN {col} {dtype}")
                print(f"  Added column: {col}")
        conn.commit()


def merge_nasa_weather(conn):
    """Merge NASA POWER weather data into daily_features."""
    weather_path = os.path.join(PROJECT_DIR, "data", "weather_nasa_power.csv")
    extra_path = os.path.join(PROJECT_DIR, "data", "weather_nasa_power_extra.csv")

    if not os.path.exists(weather_path):
        print("  NASA weather file not found, skipping")
        return

    print("  Loading NASA POWER weather...")
    weather = pd.read_csv(weather_path)
    weather["date"] = pd.to_datetime(weather["date"]).dt.date
    print(f"  Weather rows: {len(weather):,}")

    # Merge extra weather (precipitation, wind_direction)
    if os.path.exists(extra_path):
        extra = pd.read_csv(extra_path)
        extra["date"] = pd.to_datetime(extra["date"]).dt.date
        weather = weather.merge(extra, on=["date", "station_id"], how="left")
        print(f"  Merged extra weather (precipitation, wind_direction)")

    with conn.cursor() as cur:
        updated = 0
        batch_size = 5000
        rows = weather.to_dict("records")

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            for row in batch:
                cur.execute("""
                    UPDATE daily_features
                    SET nasa_temperature = %s,
                        nasa_humidity = %s,
                        nasa_wind_speed = %s,
                        precipitation = %s,
                        wind_direction = %s
                    WHERE station_id = %s AND date = %s
                      AND nasa_temperature IS NULL
                """, (
                    row.get("nasa_temperature"),
                    row.get("nasa_humidity"),
                    row.get("nasa_wind_speed"),
                    row.get("precipitation") if pd.notna(row.get("precipitation")) else None,
                    row.get("wind_direction") if pd.notna(row.get("wind_direction")) else None,
                    row["station_id"],
                    row["date"],
                ))
            conn.commit()
            updated += len(batch)
            if (i + batch_size) % 50000 == 0:
                print(f"    Weather progress: {i + batch_size:,}/{len(rows):,}")

        print(f"  NASA weather merge complete ({updated:,} rows processed)")


def merge_fire_data(conn):
    """Merge NASA FIRMS fire counts into daily_features."""
    fire_path = os.path.join(PROJECT_DIR, "data", "fire_counts_firms.csv")

    if not os.path.exists(fire_path):
        print("  Fire data file not found, skipping")
        return

    print("  Loading NASA FIRMS fire data...")
    fire = pd.read_csv(fire_path)
    fire["date"] = pd.to_datetime(fire["date"]).dt.date
    print(f"  Fire rows: {len(fire):,}")

    with conn.cursor() as cur:
        updated = 0
        for _, row in fire.iterrows():
            cur.execute("""
                UPDATE daily_features
                SET fire_count = %s
                WHERE station_id = %s AND date = %s
                  AND fire_count IS NULL
            """, (
                int(row["fire_count"]),
                int(row["station_id"]),
                row["date"],
            ))
            updated += 1
            if updated % 50000 == 0:
                conn.commit()
                print(f"    Fire progress: {updated:,}/{len(fire):,}")

        conn.commit()
        print(f"  Fire data merge complete ({updated:,} rows processed)")


def shift_rolling_features(conn):
    """Fix data leakage: shift roll_3_mean and roll_7_mean by 1 day.
    
    Current: roll_3_mean includes today's value (leakage!)
    Fixed: roll_3_mean uses only past values (shifted by 1).
    """
    print("  Fixing rolling feature leakage (shifting by 1 day)...")
    with conn.cursor() as cur:
        # For each station+parameter combo, recalculate rolling features
        # using lag values instead of including current day
        cur.execute("""
            UPDATE daily_features df
            SET roll_3_mean = (COALESCE(lag_1, 0) + COALESCE(lag_2, 0) + COALESCE(lag_3, 0)) / 
                NULLIF(
                    (CASE WHEN lag_1 IS NOT NULL THEN 1 ELSE 0 END) +
                    (CASE WHEN lag_2 IS NOT NULL THEN 1 ELSE 0 END) +
                    (CASE WHEN lag_3 IS NOT NULL THEN 1 ELSE 0 END), 0
                )
            WHERE lag_1 IS NOT NULL
        """)
        fixed = cur.rowcount
        conn.commit()
        print(f"  Fixed roll_3_mean for {fixed:,} rows (now uses lag_1+lag_2+lag_3)")


def main():
    start = time.time()
    conn = psycopg2.connect(**DB_CONFIG)

    # Show starting state
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.country_code, COUNT(DISTINCT s.id), COUNT(r.id)
            FROM stations s
            LEFT JOIN raw_measurements r ON s.id = r.station_id
            GROUP BY s.country_code ORDER BY s.country_code
        """)
        print("Starting state (raw_measurements):")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} stations, {row[2]:,} measurements")

    # Phase 1: Cleaning
    print("\n" + "=" * 60)
    print("Phase 1: Cleaning Pipeline (all countries)")
    print("=" * 60)
    clean_report = run_cleaning_pipeline(conn)
    print(f"Cleaning done: {clean_report}")

    # Phase 2: Feature Engineering
    print("\n" + "=" * 60)
    print("Phase 2: Feature Engineering (all countries)")
    print("=" * 60)
    feature_report = run_feature_pipeline(conn)
    print(f"Features done: {feature_report}")

    # Phase 3: Add country_code
    print("\n" + "=" * 60)
    print("Phase 3: Add country_code to daily_features")
    print("=" * 60)
    add_country_column(conn)

    # Phase 4: Add NASA columns and merge
    print("\n" + "=" * 60)
    print("Phase 4: Merge NASA weather + fire data")
    print("=" * 60)
    add_nasa_columns(conn)
    merge_nasa_weather(conn)
    merge_fire_data(conn)

    # Phase 5: Fix data leakage
    print("\n" + "=" * 60)
    print("Phase 5: Fix rolling feature leakage")
    print("=" * 60)
    shift_rolling_features(conn)

    # Final status
    print("\n" + "=" * 60)
    print("FINAL DATABASE STATUS")
    print("=" * 60)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM raw_measurements")
        print(f"  raw_measurements:    {cur.fetchone()[0]:,}")
        cur.execute("SELECT COUNT(*) FROM clean_measurements")
        print(f"  clean_measurements:  {cur.fetchone()[0]:,}")
        cur.execute("SELECT COUNT(*) FROM daily_features")
        print(f"  daily_features:      {cur.fetchone()[0]:,}")

        cur.execute("""
            SELECT country_code, parameter, COUNT(*),
                   SUM(CASE WHEN nasa_temperature IS NOT NULL THEN 1 ELSE 0 END) as has_weather,
                   SUM(CASE WHEN fire_count IS NOT NULL THEN 1 ELSE 0 END) as has_fire
            FROM daily_features
            GROUP BY country_code, parameter
            ORDER BY country_code, parameter
        """)
        print("\n  Breakdown:")
        print(f"  {'Country':>8} {'Param':>6} {'Rows':>10} {'Weather':>10} {'Fire':>8}")
        print(f"  {'-'*50}")
        for row in cur.fetchall():
            print(f"  {row[0] or '?':>8} {row[1]:>6} {row[2]:>10,} {row[3]:>10,} {row[4]:>8,}")

    elapsed = time.time() - start
    print(f"\nTotal time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    conn.close()


if __name__ == "__main__":
    main()
