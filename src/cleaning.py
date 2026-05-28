"""
IndiaAQ Intelligence — Data Cleaning Pipeline
==============================================
Reads from raw_measurements, applies 5-phase cleaning,
writes to clean_measurements.

Processes station-by-station to avoid memory issues (7.78M rows).

Phases:
    1. Remove NULL/zero values
    2. Remove placeholder values (999.99, 9999)
    3. Remove physically impossible negatives
    4. Remove out-of-range outliers (per-parameter thresholds)
    5. Flag and track what was removed (cleaning_flags)
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ─── Domain-Knowledge Thresholds ──────────────────────────
# Maximum physically realistic values for each parameter
THRESHOLDS = {
    "pm25": {"min": 0, "max": 500},          # Diwali peaks ~500
    "pm10": {"min": 0, "max": 600},           # Dust storms max ~600
    "no2":  {"min": 0, "max": 400},           # Industrial max ~400
    "no":   {"min": 0, "max": 400},           # Same family as no2
    "nox":  {"min": 0, "max": 500},           # NOx = NO + NO2
    "o3":   {"min": 0, "max": 300},           # Rarely exceeds
    "co":   {"min": 0, "max": 10000},         # 81,450 in data = broken
    "so2":  {"min": 0, "max": 200},           # 1,958 in data = broken
    "temperature":       {"min": -10, "max": 55},   # India range
    "relativehumidity":  {"min": 0, "max": 100},     # Physical limit
    "wind_speed":        {"min": 0, "max": 50},       # Above = cyclone
    "wind_direction":    {"min": 0, "max": 360},      # Compass degrees
    "pm1":  {"min": 0, "max": 300},
    "um003": {"min": 0, "max": 100000},       # Particle count, wide range
    "pressure": {"min": 800, "max": 1100},    # Atmospheric pressure hPa
}

# Exact placeholder values sensors report as error codes
PLACEHOLDER_VALUES = {999.99, 9999, 9999.0, -999, -999.0}


# ─── Phase 1: Remove NULL/Zero ────────────────────────────
def remove_nulls(df):
    """Remove rows where value is NULL or exactly 0 for concentration params."""
    before = len(df)
    flags = []

    # Remove NaN values
    null_mask = df["value"].isna()
    null_count = null_mask.sum()
    if null_count > 0:
        flags.append(f"null_removed:{null_count}")

    # Remove exact zeros for pollutants (not for temp/humidity/wind)
    zero_params = ["pm25", "pm10", "no2", "no", "nox", "o3", "co", "so2"]
    zero_mask = (df["parameter"].isin(zero_params)) & (df["value"] == 0)
    zero_count = zero_mask.sum()
    if zero_count > 0:
        flags.append(f"zero_removed:{zero_count}")

    # Combined mask: keep rows that are NOT null AND NOT zero-pollutant
    drop_mask = null_mask | zero_mask
    df_clean = df[~drop_mask].copy()

    return df_clean, flags


# ─── Phase 2: Remove Placeholders ─────────────────────────
def remove_placeholders(df):
    """Remove sensor error codes (999.99, 9999, etc.)."""
    before = len(df)
    mask = df["value"].isin(PLACEHOLDER_VALUES)
    count = mask.sum()
    flags = []

    if count > 0:
        flags.append(f"placeholder_removed:{count}")

    return df[~mask].copy(), flags


# ─── Phase 3: Remove Impossible Negatives ─────────────────
def remove_negatives(df):
    """Remove negative values for parameters that can't be negative."""
    # Parameters that CAN be negative: temperature (but limited)
    # Everything else: concentrations, humidity, wind = must be >= 0
    non_negative_params = [
        "pm25", "pm10", "no2", "no", "nox", "o3", "co", "so2",
        "relativehumidity", "wind_speed", "wind_direction",
        "pm1", "um003",
    ]

    mask = (df["parameter"].isin(non_negative_params)) & (df["value"] < 0)
    count = mask.sum()
    flags = []

    if count > 0:
        flags.append(f"negative_removed:{count}")

    return df[~mask].copy(), flags


# ─── Phase 4: Remove Out-of-Range Outliers ────────────────
def remove_outliers(df):
    """Remove values outside domain-knowledge thresholds."""
    flags = []
    total_removed = 0
    mask = pd.Series(False, index=df.index)

    for param, limits in THRESHOLDS.items():
        param_mask = (
            (df["parameter"] == param) &
            ((df["value"] < limits["min"]) | (df["value"] > limits["max"]))
        )
        count = param_mask.sum()
        if count > 0:
            flags.append(f"outlier_{param}:{count}")
            total_removed += count
        mask = mask | param_mask

    return df[~mask].copy(), flags


# ─── Full Cleaning Pipeline ──────────────────────────────
def clean_station_data(df):
    """
    Run all 4 cleaning phases on a station's data.

    Args:
        df: DataFrame with columns [station_id, sensor_id, parameter,
            value, unit, datetime_utc, datetime_local]

    Returns:
        (cleaned_df, all_flags, report_dict)
    """
    original_count = len(df)
    all_flags = []

    # Phase 1: Nulls and zeros
    df, flags = remove_nulls(df)
    all_flags.extend(flags)

    # Phase 2: Placeholders
    df, flags = remove_placeholders(df)
    all_flags.extend(flags)

    # Phase 3: Negatives
    df, flags = remove_negatives(df)
    all_flags.extend(flags)

    # Phase 4: Outliers
    df, flags = remove_outliers(df)
    all_flags.extend(flags)

    # Add cleaning metadata
    df["cleaning_flags"] = str(all_flags) if all_flags else None
    df["is_valid"] = True

    final_count = len(df)
    removed = original_count - final_count
    pct = (removed / original_count * 100) if original_count > 0 else 0

    report = {
        "original": original_count,
        "final": final_count,
        "removed": removed,
        "removed_pct": round(pct, 2),
        "flags": all_flags,
    }

    return df, all_flags, report


# ─── Database Operations ─────────────────────────────────
def load_station_raw_data(conn, station_id):
    """Load all raw measurements for one station."""
    sql = """
        SELECT station_id, sensor_id, parameter, value, unit,
               datetime_utc, datetime_local
        FROM raw_measurements
        WHERE station_id = %s
        ORDER BY datetime_utc
    """
    return pd.read_sql(sql, conn, params=(station_id,))


def insert_clean_data(conn, df):
    """Bulk insert cleaned data into clean_measurements."""
    if df.empty:
        return 0

    sql = """
        INSERT INTO clean_measurements
            (station_id, sensor_id, parameter, value, unit,
             datetime_utc, datetime_local, cleaning_flags, is_valid)
        VALUES %s
        ON CONFLICT (station_id, parameter, datetime_utc) DO NOTHING
    """
    values = [
        (
            row.station_id, row.sensor_id, row.parameter,
            row.value, row.unit, row.datetime_utc, row.datetime_local,
            row.cleaning_flags if isinstance(row.cleaning_flags, list)
            else [row.cleaning_flags] if row.cleaning_flags else [],
            row.is_valid,
        )
        for row in df.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    return len(values)


def get_all_station_ids(conn):
    """Get list of all station IDs that have raw data."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT station_id
            FROM raw_measurements
            ORDER BY station_id
        """)
        return [row[0] for row in cur.fetchall()]


# ─── Main Entry Point ────────────────────────────────────
def run_cleaning_pipeline(conn, station_ids=None):
    """
    Run cleaning on all (or specified) stations.

    Args:
        conn: psycopg2 connection
        station_ids: list of station IDs to process (None = all)

    Returns:
        Summary report dict
    """
    if station_ids is None:
        station_ids = get_all_station_ids(conn)

    total_stations = len(station_ids)
    total_original = 0
    total_cleaned = 0
    total_removed = 0

    for i, sid in enumerate(station_ids):
        # Load raw data
        df = load_station_raw_data(conn, sid)
        if df.empty:
            continue

        # Clean
        df_clean, flags, report = clean_station_data(df)

        # Insert
        inserted = insert_clean_data(conn, df_clean)

        # Track totals
        total_original += report["original"]
        total_cleaned += report["final"]
        total_removed += report["removed"]

        print(
            f"  [{i+1}/{total_stations}] Station {sid}: "
            f"{report['original']} → {report['final']} "
            f"(-{report['removed']}, {report['removed_pct']}%)"
        )

    summary = {
        "stations_processed": total_stations,
        "total_original": total_original,
        "total_cleaned": total_cleaned,
        "total_removed": total_removed,
        "removed_pct": round(total_removed / total_original * 100, 2) if total_original > 0 else 0,
    }

    print(f"\n  ✅ Cleaning complete:")
    print(f"     Original:  {summary['total_original']:,}")
    print(f"     Cleaned:   {summary['total_cleaned']:,}")
    print(f"     Removed:   {summary['total_removed']:,} ({summary['removed_pct']}%)")

    return summary
