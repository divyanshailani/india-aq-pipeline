"""
Global AQ Intelligence — Daily ETL Pipeline (V2 Batch)
=======================================================
Orchestrates: raw_measurements → cleaning → clean_measurements → features → daily_features

V2 (2026-06-28): All phases use batch operations.
    - Phase 1 (Cleaning): ONE bulk load → vectorized clean → ONE bulk insert
    - Phase 2 (Features): ONE bulk load → in-memory groupby → ONE bulk upsert
    - Phase 3 (Weather):  Concurrent API fetching → batch DB update
    - Phase 4 (Advanced):  Single SQL window function (unchanged, already efficient)

Usage:
    python scripts/run_daily_etl.py                    # daily mode: only recent stations
    python scripts/run_daily_etl.py --all              # process ALL stations (slow)
    python scripts/run_daily_etl.py --station-id 1     # process one station
    python scripts/run_daily_etl.py --clean-only       # only run cleaning
    python scripts/run_daily_etl.py --features-only    # only run features
"""

import sys
import os
import argparse
import time
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path so we can import src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cleaning import run_cleaning_pipeline
from src.features import run_feature_pipeline
from src.config import DB_CONFIG


def get_recent_station_ids(conn, days=3):
    """Get station IDs that have received new raw data in the last N days."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT station_id
            FROM raw_measurements
            WHERE datetime_utc >= NOW() - INTERVAL '%s days'
            ORDER BY station_id
        """, (days,))
        ids = [row[0] for row in cur.fetchall()]
    return ids


def ensure_weather_columns(conn):
    """Ensure weather and AOD columns exist before fetching."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE daily_features
                ADD COLUMN IF NOT EXISTS om_temperature DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS om_wind_speed DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS om_precipitation DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS om_aerosol_optical_depth DOUBLE PRECISION
        """)
    conn.commit()

def main():
    parser = argparse.ArgumentParser(description="Global AQ Daily ETL Pipeline")
    parser.add_argument("--station-id", type=int, default=None,
                        help="Process only this station ID")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL stations (slow, use for full rebuild)")
    parser.add_argument("--recent-days", type=int, default=3,
                        help="Days of recent data to process (default: 3)")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only run cleaning (skip features)")
    parser.add_argument("--features-only", action="store_true",
                        help="Only run features (skip cleaning)")
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)
    ensure_weather_columns(conn)

    # Determine which stations to process
    if args.station_id:
        station_ids = [args.station_id]
        print(f"📌 Single station mode: station_id={args.station_id}")
    elif args.all:
        station_ids = None  # None = all stations
        print("🔄 FULL REBUILD MODE: processing ALL stations (this will take a while)")
    else:
        station_ids = get_recent_station_ids(conn, days=args.recent_days)
        print(f"📡 Daily mode: {len(station_ids)} stations with data in last {args.recent_days} days")
        if not station_ids:
            print("  No stations with recent data — nothing to process.")
            conn.close()
            return

    start_time = time.time()

    # ── Step 1: Cleaning (V2 Batch) ──
    if not args.features_only:
        print("=" * 50)
        print("🧹 Phase 1: Cleaning Pipeline (V2 Batch)")
        print("=" * 50)
        clean_report = run_cleaning_pipeline(
            conn, station_ids,
            recent_days=args.recent_days + 1  # +1 buffer for timezone edge
        )
        print()

    # ── Step 2: Feature Engineering (V2 Batch) ──
    if not args.clean_only:
        print("=" * 50)
        print("🛠️  Phase 2: Feature Engineering (V2 Batch)")
        print("=" * 50)
        feature_report = run_feature_pipeline(
            conn, station_ids,
            lookback_days=90,                  # 90 days context for lag_30 + rolling_30
            insert_recent_days=args.recent_days + 4  # insert window = recent + buffer
        )
        print()

    # ── Step 3: Concurrent Weather & AOD Enrichment ──
    if not args.clean_only:
        print("=" * 50)
        print("🌍 Phase 3: Weather & AOD Enrichment (Concurrent)")
        print("=" * 50)

        from src.api_fallback_manager import ApiFallbackManager
        from scripts.fetch_daily_weather import fetch_weather_for_date
        from scripts.fetch_daily_aod import fetch_aod_for_date

        import re
        raw_keys = os.getenv("OPENAQ_KEYS", "")
        raw_keys = re.sub(r'[\r\n]+', ',', raw_keys)  # normalize newlines to commas
        clean_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

        fallback_manager = ApiFallbackManager(
            openaq_keys=clean_keys,
            max_retries=3,
            base_backoff=2.0
        )

        with conn.cursor() as cur:
            # Query for rows missing weather data
            if station_ids is None:
                cur.execute("""
                    SELECT DISTINCT df.station_id, df.date, s.latitude, s.longitude
                    FROM daily_features df
                    JOIN stations s ON df.station_id = s.id
                    WHERE df.om_temperature IS NULL
                       OR df.om_precipitation IS NULL
                    ORDER BY df.date DESC, df.station_id
                """)
            else:
                format_strings = ','.join(['%s'] * len(station_ids))
                cur.execute(f"""
                    SELECT DISTINCT df.station_id, df.date, s.latitude, s.longitude
                    FROM daily_features df
                    JOIN stations s ON df.station_id = s.id
                    WHERE df.station_id IN ({format_strings})
                      AND (df.om_temperature IS NULL
                           OR df.om_precipitation IS NULL)
                    ORDER BY df.date DESC, df.station_id
                """, tuple(station_ids))

            missing_rows = cur.fetchall()

        if not missing_rows:
            print("  ✓ All daily_features have complete weather & AOD context.")
        else:
            print(f"  Fetching Weather & AOD for {len(missing_rows)} rows (concurrent)...")

            # ── Concurrent API fetching with ThreadPoolExecutor ──
            def fetch_single(row_data):
                """Fetch weather + AOD for a single station/date pair."""
                sid, dt, lat, lon = row_data
                target_date_str = dt.strftime("%Y-%m-%d")
                w_data = fetch_weather_for_date(fallback_manager, lat, lon, target_date_str)
                aod_data = fetch_aod_for_date(fallback_manager, lat, lon, target_date_str)
                return (
                    w_data["om_temperature"],
                    w_data["om_wind_speed"],
                    w_data["om_precipitation"],
                    aod_data["om_aerosol_optical_depth"],
                    sid,
                    dt
                )

            successful_updates = []
            failed_rows = []

            # Use 5 concurrent workers (gentle on Open-Meteo rate limits)
            max_workers = min(5, len(missing_rows))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(fetch_single, row): row
                    for row in missing_rows
                }

                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    sid, dt = row[0], row[1]
                    try:
                        result = future.result()
                        successful_updates.append(result)
                    except Exception as e:
                        print(f"  ⚠️ Failed: Station {sid} on {dt}: {e}")
                        failed_rows.append((sid, dt))

            # ── Batch UPDATE all successful results ──
            if successful_updates:
                print(f"  💾 Batch updating {len(successful_updates)} weather rows...")
                with conn.cursor() as cur:
                    # Use a temp-table-free approach: batch execute individual UPDATEs
                    # This is still much faster than commit-per-row
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, """
                        UPDATE daily_features
                        SET om_temperature = %s,
                            om_wind_speed = %s,
                            om_precipitation = %s,
                            om_aerosol_optical_depth = %s
                        WHERE station_id = %s AND date = %s
                    """, successful_updates, page_size=500)
                conn.commit()
                print(f"  ✅ Updated {len(successful_updates)} rows.")

            # ── Batch DELETE failed rows (atomic abort) ──
            if failed_rows:
                print(f"  🚨 Rolling back {len(failed_rows)} failed rows...")
                with conn.cursor() as cur:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, """
                        DELETE FROM daily_features
                        WHERE station_id = %s AND date = %s
                    """, failed_rows, page_size=500)
                conn.commit()
                print(f"  Deleted {len(failed_rows)} corrupted rows.")

        print()

    # ── Step 4: Advanced Weather Engineering ──
    if not args.clean_only:
        print("=" * 50)
        print("⚡ Phase 4: Advanced Weather Engineering")
        print("=" * 50)
        from src.features import build_advanced_weather_features
        updated_rows = build_advanced_weather_features(conn)
        print(f"  ✅ Computed rolling_3day_precip & aod_volatility_index for {updated_rows:,} rows.")
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

