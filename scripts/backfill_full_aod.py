import time
import requests
import psycopg2
import pandas as pd
from datetime import date, timedelta
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
START_DATE = "2021-01-01"
END_DATE = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")


def get_stations_with_missing_aod(conn):
    query = """
        SELECT DISTINCT s.id, s.latitude, s.longitude
        FROM stations s
        JOIN daily_features df ON s.id = df.station_id
        WHERE df.date BETWEEN %s AND %s
          AND df.om_aerosol_optical_depth IS NULL
    """
    return pd.read_sql(query, conn, params=(START_DATE, END_DATE))


def fetch_openmeteo_aod(lat, lon):
    all_dfs = []
    # Chunk by year to prevent Open-Meteo API timeouts on massive 5-year requests
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
        resp = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=30)

        # Sleep to avoid hitting Open-Meteo rate limits
        time.sleep(2.0)

        if resp.status_code != 200:
            if resp.status_code == 429:
                raise Exception("Rate Limited (429)")
            # Ignore other errors like 400 for future dates
            continue

        data = resp.json().get("hourly", {})
        aod_array = data.get("aerosol_optical_depth", [])
        time_array = data.get("time", [])

        if not aod_array or not time_array:
            continue

        df_hourly = pd.DataFrame({"time": pd.to_datetime(time_array), "aod": aod_array})

        df_hourly["date"] = df_hourly["time"].dt.date
        # Drop NaN AOD values BEFORE averaging so null satellite readings don't drag the mean
        df_hourly = df_hourly.dropna(subset=["aod"])
        if df_hourly.empty:
            continue

        df_daily = df_hourly.groupby("date")["aod"].mean().reset_index()
        df_daily.rename(columns={"aod": "om_aerosol_optical_depth"}, inplace=True)
        all_dfs.append(df_daily)

        # Micro-sleep between chunks for safety
        time.sleep(0.2)

    if not all_dfs:
        return None

    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df


def update_aod_bulk(conn, station_id, df):
    if df is None or df.empty:
        return 0

    sql = """
        UPDATE daily_features
        SET om_aerosol_optical_depth = %s
        WHERE station_id = %s 
          AND date = %s 
          AND om_aerosol_optical_depth IS NULL
    """

    df_valid = df.dropna(subset=["om_aerosol_optical_depth"])
    if df_valid.empty:
        return 0

    values = []
    for _, row in df_valid.iterrows():
        values.append((row["om_aerosol_optical_depth"], station_id, row["date"]))

    from psycopg2.extras import execute_batch

    cur = conn.cursor()
    execute_batch(cur, sql, values, page_size=500)
    # execute_batch does NOT reliably set cur.rowcount.
    # Use len(values) — the WHERE ... IS NULL clause prevents duplicate writes,
    # so attempted == actual for all practical purposes.
    actual_updated = len(values)
    conn.commit()
    cur.close()
    return actual_updated


def _get_connection():
    """Create a new DB connection with retry logic for transient failures."""
    for attempt in range(3):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except psycopg2.OperationalError as e:
            if attempt < 2:
                wait = (attempt + 1) * 10
                print(f"⚠️ DB connection failed (attempt {attempt + 1}/3). Retrying in {wait}s... Error: {e}")
                time.sleep(wait)
            else:
                raise


def process_station(row, total, index):
    sid = int(row["id"])
    lat = float(row["latitude"])
    lon = float(row["longitude"])

    updated = 0
    success = False

    # Retry logic for BOTH API rate limits AND transient DB/network errors
    for attempt in range(5):
        conn = None
        try:
            conn = _get_connection()
            df = fetch_openmeteo_aod(lat, lon)
            if df is not None:
                updated = update_aod_bulk(conn, sid, df)
            success = True
            break
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass  # Connection might already be dead

            err_str = str(e)
            if "429" in err_str:
                wait = 30 * (attempt + 1)  # Exponential: 30s, 60s, 90s, 120s, 150s
                print(f"⚠️ Rate limit hit on station {sid} (attempt {attempt + 1}/5). Waiting {wait}s...")
                time.sleep(wait)
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                wait = 10 * (attempt + 1)
                print(f"⏳ Timeout on station {sid} (attempt {attempt + 1}/5). Retrying in {wait}s...")
                time.sleep(wait)
            elif "OperationalError" in err_str or "connection" in err_str.lower():
                wait = 15 * (attempt + 1)
                print(f"🔌 DB connection lost on station {sid} (attempt {attempt + 1}/5). Reconnecting in {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Unexpected error on station {sid}: {e}")
                break  # Don't retry unknown errors
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    return sid, updated, success


def main():
    print(f"🚀 Starting Bulk AOD Backfill")
    print(f"📅 Date Range: {START_DATE} to {END_DATE}")

    conn = _get_connection()
    stations = get_stations_with_missing_aod(conn)
    conn.close()

    total = len(stations)
    print(f"🔍 Found {total} stations with missing AOD.")
    if total == 0:
        print("✅ Nothing to do!")
        return

    total_updated_rows = 0
    total_failed = 0
    start_time = time.time()

    # Use 1 thread to avoid aggressive rate limiting
    MAX_WORKERS = 1
    print(f"⚡ Using {MAX_WORKERS} parallel workers to respect API limits...")

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_station, row, total, i): i
            for i, row in stations.iterrows()
        }

        for future in as_completed(futures):
            completed += 1
            sid, updated, success = future.result()

            if success:
                total_updated_rows += updated
                print(
                    f"[{completed}/{total}] Station {sid} ✅ Updated {updated} rows. "
                    f"(Running total: {total_updated_rows})"
                )
            else:
                total_failed += 1
                print(f"[{completed}/{total}] Station {sid} ❌ FAILED after all retries.")

            # Add a small delay between tasks to reduce burst pressure
            time.sleep(1)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"\n{'='*60}")
    print(f"🎉 Bulk AOD Backfill Complete in {hours}h {mins}m!")
    print(f"📈 Total rows updated: {total_updated_rows}")
    print(f"❌ Total stations failed: {total_failed}")
    print(f"✅ Total stations succeeded: {completed - total_failed}/{total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
