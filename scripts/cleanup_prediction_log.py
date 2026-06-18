"""
Clean impossible prediction_log rows.

Rows where target_date < run_date were generated from stale station anchors.
They are invalid for forward validation, so this script archives them first and
then deletes them from prediction_log.

Usage:
    python scripts/cleanup_prediction_log.py --dry-run
    python scripts/cleanup_prediction_log.py --execute
"""

import argparse
import os
import sys

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG


def ensure_archive_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_log_archive
            (LIKE prediction_log INCLUDING DEFAULTS INCLUDING CONSTRAINTS)
        """)
        cur.execute("""
            ALTER TABLE prediction_log_archive
                ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP DEFAULT now(),
                ADD COLUMN IF NOT EXISTS archive_reason TEXT
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_log_archive_reason
                ON prediction_log_archive (archive_reason, archived_at)
        """)
    conn.commit()


def summarize_bad_rows(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) AS total,
                   MIN(target_date) AS min_target,
                   MAX(target_date) AS max_target
            FROM prediction_log
            WHERE target_date < run_date
        """)
        total, min_target, max_target = cur.fetchone()

        cur.execute("""
            SELECT country_code, COUNT(*)
            FROM prediction_log
            WHERE target_date < run_date
            GROUP BY country_code
            ORDER BY country_code
        """)
        by_country = cur.fetchall()

    return total, min_target, max_target, by_country


def archive_and_delete(conn):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO prediction_log_archive (
                id, run_id, run_date, country_code, station_id, target_date,
                horizon_days, predicted_value, actual_value, error,
                validated_at, created_at, archive_reason
            )
            SELECT
                id, run_id, run_date, country_code, station_id, target_date,
                horizon_days, predicted_value, actual_value, error,
                validated_at, created_at, 'target_date_before_run_date'
            FROM prediction_log
            WHERE target_date < run_date
        """)
        archived = cur.rowcount

        cur.execute("""
            DELETE FROM prediction_log
            WHERE target_date < run_date
        """)
        deleted = cur.rowcount

    conn.commit()
    return archived, deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show rows that would be cleaned")
    parser.add_argument("--execute", action="store_true", help="Archive and delete impossible rows")
    args = parser.parse_args()

    if args.dry_run == args.execute:
        parser.error("Choose exactly one: --dry-run or --execute")

    conn = psycopg2.connect(**DB_CONFIG)
    ensure_archive_table(conn)

    total, min_target, max_target, by_country = summarize_bad_rows(conn)
    print("Impossible prediction rows: target_date < run_date")
    print(f"  total:      {total:,}")
    print(f"  date range: {min_target} -> {max_target}")
    print("  by country:")
    for country_code, count in by_country:
        print(f"    {country_code}: {count:,}")

    if args.execute:
        archived, deleted = archive_and_delete(conn)
        print("\nCleanup complete")
        print(f"  archived: {archived:,}")
        print(f"  deleted:  {deleted:,}")
    else:
        print("\nDry run only. Re-run with --execute to archive and delete.")

    conn.close()


if __name__ == "__main__":
    main()
