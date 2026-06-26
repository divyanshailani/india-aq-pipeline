"""
Multi-country daily data collector with chunked backfill.
Runs nightly via launchd. Does TWO things:

1. INCREMENTAL: Fetch last 7 days for all countries (keep data fresh)
2. BACKFILL:    Fetch next 90-day chunk of historical data (2021 → present)

The backfill advances automatically each night. After ~22 nights,
all countries will have complete 2021-present data.

Backfill progress is tracked in: logs/backfill_state.json

Usage:
    python scripts/run_daily_collector.py                   # normal nightly run
    python scripts/run_daily_collector.py --incremental-only # skip backfill
    python scripts/run_daily_collector.py --backfill-only    # skip incremental
    python scripts/run_daily_collector.py --country US       # single country
    python scripts/run_daily_collector.py --chunk-days 30    # smaller chunks
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.fetch_openaq import run_fetch, COUNTRIES

# Paths
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "collection_log.json")
BACKFILL_STATE_FILE = os.path.join(LOG_DIR, "backfill_state.json")

BACKFILL_START = "2021-01-01"
# Countries to backfill (skip IN — already done)
BACKFILL_COUNTRIES = ["US", "GB", "AU"]


# Logging
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {"runs": []}


def save_log(log):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2, default=str)


# Backfill state
def load_backfill_state():
    """Load backfill progress for each country."""
    if os.path.exists(BACKFILL_STATE_FILE):
        with open(BACKFILL_STATE_FILE, "r") as f:
            return json.load(f)

    # Initialize: all countries start from 2021-01-01
    state = {}
    for cc in BACKFILL_COUNTRIES:
        state[cc] = {
            "current_start": BACKFILL_START,
            "status": "in_progress",
        }
    return state


def save_backfill_state(state):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(BACKFILL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


from src.config import DB_CONFIG
import psycopg2

def get_gap_days(cc):
    """Query the DB for the most recent data point for this country to calculate exact days to fetch."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MAX(date) 
                FROM daily_features
                WHERE country_code = %s
            """, (cc,))
            row = cur.fetchone()
        conn.close()
        
        if row and row[0]:
            last_dt = row[0]
            if isinstance(last_dt, datetime):
                last_dt = last_dt.replace(tzinfo=None).date()
            elif hasattr(last_dt, 'date'):
                last_dt = last_dt.date()
            
            gap = (datetime.utcnow().date() - last_dt).days
            return max(1, gap + 1) # fetch at least 1 day, plus 1 for safe overlap
    except Exception as e:
        print(f"  Warning: could not compute gap for {cc} ({e}), defaulting to 7")
    return 7


def run_incremental(countries, days=7):
    """Fetch recent data for all countries."""
    print(f"\n  PHASE 1: Incremental")
    print(f"  {'-'*50}")

    results = {}
    for cc in countries:
        try:
            # Dynamically calculate exact gap to avoid over-fetching and wasting time.
            # Use 'days' argument only if explicitly overridden from the default 7.
            fetch_days = get_gap_days(cc) if days == 7 else days
            
            print(f"\n  [INCREMENTAL] {cc} gap calculated. Fetching last {fetch_days} days...")
            stats = run_fetch(cc, days=fetch_days)
            results[cc] = {"status": "success", "rows": stats["rows_inserted"]}
        except Exception as e:
            print(f"  {cc} FAILED: {e}")
            results[cc] = {"status": "failed", "error": str(e)}

    return results


# Backfill fetch
def run_backfill(countries, chunk_days=90):
    """
    Fetch the next chunk of historical data for each country.
    Advances the window by chunk_days each night.
    """
    print(f"\n  PHASE 2: Backfill ({chunk_days}-day chunks)")
    print(f"  {'-'*50}")

    state = load_backfill_state()
    today = datetime.now().strftime("%Y-%m-%d")
    results = {}

    for cc in countries:
        if cc not in state:
            state[cc] = {"current_start": BACKFILL_START, "status": "in_progress"}

        entry = state[cc]

        if entry["status"] == "complete":
            print(f"  {cc}: Backfill already complete")
            results[cc] = {"status": "complete"}
            continue

        chunk_start = entry["current_start"]
        chunk_end_dt = datetime.strptime(chunk_start, "%Y-%m-%d") + timedelta(days=chunk_days)
        chunk_end = chunk_end_dt.strftime("%Y-%m-%d")

        # If chunk end is past today, cap it and mark complete after this run
        is_final_chunk = chunk_end >= today
        if is_final_chunk:
            chunk_end = today

        print(f"\n  {cc}: Backfilling {chunk_start} to {chunk_end}")

        try:
            stats = run_fetch(cc, date_from=chunk_start, date_to=chunk_end)
            results[cc] = {
                "status": "success",
                "chunk": f"{chunk_start} to {chunk_end}",
                "rows": stats["rows_inserted"],
            }

            # Advance the window
            if is_final_chunk:
                entry["status"] = "complete"
                entry["completed_at"] = today
                print(f"  {cc}: BACKFILL COMPLETE!")
            else:
                entry["current_start"] = chunk_end

        except Exception as e:
            print(f"  {cc} FAILED: {e}")
            results[cc] = {"status": "failed", "error": str(e)}
            # Don't advance — retry same chunk next time

    save_backfill_state(state)

    # Print progress summary
    print(f"\n  Backfill progress:")
    for cc, entry in state.items():
        if entry["status"] == "complete":
            print(f"    {cc}: COMPLETE (finished {entry.get('completed_at', '?')})")
        else:
            current = entry["current_start"]
            days_done = (datetime.strptime(current, "%Y-%m-%d") -
                        datetime.strptime(BACKFILL_START, "%Y-%m-%d")).days
            total_days = (datetime.now() -
                         datetime.strptime(BACKFILL_START, "%Y-%m-%d")).days
            pct = min(100, int(days_done / total_days * 100))
            print(f"    {cc}: {pct}% ({current} / {today})")

    return results


# Main
def main():
    parser = argparse.ArgumentParser(description="Daily AQ Data Collector")
    parser.add_argument("--incremental-only", action="store_true",
                        help="Only run incremental fetch (skip backfill)")
    parser.add_argument("--backfill-only", action="store_true",
                        help="Only run backfill (skip incremental)")
    parser.add_argument("--country", type=str, default=None,
                        choices=list(COUNTRIES.keys()),
                        help="Single country (default: all)")
    parser.add_argument("--days", type=int, default=7,
                        help="Incremental: fetch last N days (default: 7)")
    parser.add_argument("--chunk-days", type=int, default=90,
                        help="Backfill: days per chunk (default: 90)")
    args = parser.parse_args()

    all_countries = [args.country] if args.country else list(COUNTRIES.keys())
    backfill_countries = [args.country] if args.country else BACKFILL_COUNTRIES

    print(f"\n{'='*60}")
    print(f"  Daily AQ Collector")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Incremental: {', '.join(all_countries)} (last {args.days} days)")
    print(f"  Backfill: {', '.join(backfill_countries)} ({args.chunk_days}-day chunks)")
    print(f"{'='*60}")

    log = load_log()
    run_entry = {
        "timestamp": datetime.now().isoformat(),
        "incremental": {},
        "backfill": {},
    }

    # Phase 1: Incremental
    if not args.backfill_only:
        run_entry["incremental"] = run_incremental(all_countries, args.days)

    # Phase 2: Backfill
    if not args.incremental_only:
        run_entry["backfill"] = run_backfill(backfill_countries, args.chunk_days)

    log["runs"].append(run_entry)
    log["runs"] = log["runs"][-30:]
    save_log(log)

    print(f"\n{'='*60}")
    print(f"  Collection complete")
    print(f"  Log: {LOG_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
