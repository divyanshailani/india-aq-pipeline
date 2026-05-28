"""
IndiaAQ Intelligence — Daily ETL Pipeline
==========================================
Orchestrates: raw_measurements → cleaning → clean_measurements → features → daily_features

Usage:
    python scripts/run_daily_etl.py                    # process all stations
    python scripts/run_daily_etl.py --station-id 1     # process one station
    python scripts/run_daily_etl.py --clean-only        # only run cleaning
    python scripts/run_daily_etl.py --features-only     # only run features
"""

import sys
import os
import argparse
import time
import psycopg2

# Add project root to path so we can import src/
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


def main():
    parser = argparse.ArgumentParser(description="IndiaAQ Daily ETL Pipeline")
    parser.add_argument("--station-id", type=int, default=None,
                        help="Process only this station ID")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only run cleaning (skip features)")
    parser.add_argument("--features-only", action="store_true",
                        help="Only run features (skip cleaning)")
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)
    station_ids = [args.station_id] if args.station_id else None

    start_time = time.time()

    # ── Step 1: Cleaning ──
    if not args.features_only:
        print("=" * 50)
        print("🧹 Phase 1: Cleaning Pipeline")
        print("=" * 50)
        clean_report = run_cleaning_pipeline(conn, station_ids)
        print()

    # ── Step 2: Feature Engineering ──
    if not args.clean_only:
        print("=" * 50)
        print("🛠️  Phase 2: Feature Engineering")
        print("=" * 50)
        feature_report = run_feature_pipeline(conn, station_ids)
        print()

    # ── Summary ──
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("=" * 50)
    print(f"✅ ETL Pipeline Complete! ({minutes}m {seconds}s)")
    print("=" * 50)

    # Show database status
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM raw_measurements")
        raw_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM clean_measurements")
        clean_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM daily_features")
        feature_count = cur.fetchone()[0]

    print(f"\n📊 Database Status:")
    print(f"   raw_measurements:    {raw_count:,}")
    print(f"   clean_measurements:  {clean_count:,}")
    print(f"   daily_features:      {feature_count:,}")

    conn.close()


if __name__ == "__main__":
    main()
