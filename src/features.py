"""
Global AQ Intelligence — Feature Engineering Pipeline (V2 Batch)
================================================================
Reads from clean_measurements, engineers features,
writes to daily_features.

V2 (2026-06-28): Eliminated sequential station-by-station DB loops.
    - ONE bulk SQL query to load all clean data (with lookback window)
    - In-memory groupby for per-station feature computation (zero network)
    - ONE bulk upsert via execute_values with page_size=5000
    - Insert window: only upsert features for recent days to avoid
      corrupting older features computed from fuller history
    - Result: 3+ hours → ~60 seconds on remote GH Actions runner

Features per station per day:
    - Time:     month, day_of_week, is_weekend, day_of_year
    - Lag:      lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_30
    - Rolling:  roll_3_mean, roll_7_mean, roll_3_std, roll_14_mean, roll_30_mean, roll_14_std
    - Momentum: pm25_delta_1, pm25_delta_7
    - Cross:    temperature, humidity, wind_speed, no2, co, o3, so2
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import warnings
from datetime import datetime, timedelta


# ─── Step 1: Load Clean Data For Station ──────────────────
def load_station_clean_data(conn, station_id):
    """Load all clean measurements for one station."""
    sql = """
        SELECT parameter, value, datetime_local
        FROM clean_measurements
        WHERE station_id = %s AND is_valid = true
        ORDER BY datetime_local
    """
    return pd.read_sql(sql, conn, params=(station_id,))


# ─── Step 2: Pivot to Daily Wide Format ───────────────────
def make_daily_wide(df):
    """
    Convert long format (one row per reading) to wide daily format
    (one row per day, one column per parameter).

    Input:
        datetime_local | parameter | value
        2024-01-15     | pm25      | 156.3
        2024-01-15     | temp      | 22.5

    Output:
        date       | pm25  | pm10 | temperature | ...
        2024-01-15 | 156.3 | 210  | 22.5        | ...
    """
    df["date"] = pd.to_datetime(df["datetime_local"]).dt.date

    # Pivot: one column per parameter, daily mean
    daily = df.pivot_table(
        index="date",
        columns="parameter",
        values="value",
        aggfunc="mean"
    )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    return daily


# ─── Step 3: Add Time Features ────────────────────────────
def add_time_features(daily):
    """Add calendar-based features."""
    daily["month"] = daily.index.month
    daily["day_of_week"] = daily.index.dayofweek     # 0=Mon, 6=Sun
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["day_of_year"] = daily.index.dayofyear
    return daily


# ─── Step 4: Add Lag Features ─────────────────────────────
def add_lag_features(daily, target_col, lags=[1, 2, 3, 7, 14, 21, 30]):
    """
    Add lagged versions of the target column.
    lag_1 = yesterday's value, lag_7 = last week's value.
    """
    for lag in lags:
        daily[f"lag_{lag}"] = daily[target_col].shift(lag)
    return daily


# ─── Step 5: Add Rolling Features ─────────────────────────
def add_rolling_features(daily, target_col, windows=[3, 7, 14, 30]):
    """
    Add rolling mean and std for the target column.
    IMPORTANT: Shift by 1 day BEFORE rolling to prevent data leakage.
    Without shift, roll_3_mean includes today's value (the prediction target).
    """
    shifted = daily[target_col].shift(1)  # exclude current day
    for w in windows:
        daily[f"roll_{w}_mean"] = shifted.rolling(w).mean()
    # Rolling std for volatility context at key windows
    daily["roll_3_std"]  = shifted.rolling(3).std()
    daily["roll_14_std"] = shifted.rolling(14).std()
    return daily


# ─── Step 5b: Add Momentum / Delta Features ───────────────
def add_delta_features(daily, target_col):
    """
    Add first-difference momentum features — all lag-shifted, no leakage.

    pm25_delta_1 = short-term momentum: yesterday vs the day before
    pm25_delta_7 = weekly momentum: yesterday vs last week
    Positive = rising pollution, negative = falling.
    """
    lag1 = daily[target_col].shift(1)
    lag2 = daily[target_col].shift(2)
    lag7 = daily[target_col].shift(7)
    daily["pm25_delta_1"] = lag1 - lag2
    daily["pm25_delta_7"] = lag1 - lag7
    return daily


# ─── Step 6: Extract Cross-Parameter Features ────────────
def extract_cross_features(daily):
    """
    Pull other parameters as features for PM2.5/PM10 prediction.
    These become input features, not targets.
    """
    cross_params = {
        "temperature": "temperature",
        "relativehumidity": "humidity",
        "wind_speed": "wind_speed",
        "no2": "no2_value",
        "co": "co_value",
        "o3": "o3_value",
        "so2": "so2_value",
    }

    result = {}
    for param_name, feature_name in cross_params.items():
        if param_name in daily.columns:
            result[feature_name] = daily[param_name]
        else:
            result[feature_name] = None

    return result


# ═══════════════════════════════════════════════════════════
# V2 BATCH FEATURE BUILDER (works on pre-loaded data)
# ═══════════════════════════════════════════════════════════

def build_features_from_data(station_df, station_id, target_param="pm25"):
    """
    Build features for one station from PRE-LOADED data.
    Zero database calls — pure in-memory pandas operations.

    This is the V2 replacement for build_features_for_station().
    Instead of loading data from the DB, it takes a pre-loaded DataFrame.

    Args:
        station_df: DataFrame with columns [parameter, value, datetime_local]
                    (already filtered to one station)
        station_id: internal station ID
        target_param: which parameter to predict ('pm25' or 'pm10')

    Returns:
        DataFrame with features, or None if insufficient data
    """
    if station_df.empty:
        return None

    # Need at least the target parameter
    if target_param not in station_df["parameter"].values:
        return None

    # Pivot to daily wide format
    daily = make_daily_wide(station_df)

    if target_param not in daily.columns:
        return None

    # Need at least 14 days of data for meaningful features
    if len(daily) < 14:
        return None

    # Add time features
    daily = add_time_features(daily)

    # Add lag features for target
    daily = add_lag_features(daily, target_param)

    # Add rolling features for target
    daily = add_rolling_features(daily, target_param)

    # Add momentum features (delta / first-difference)
    daily = add_delta_features(daily, target_param)

    # Extract cross-parameter features
    cross = extract_cross_features(daily)

    # Build final feature table
    features = pd.DataFrame(index=daily.index)
    features["date"] = daily.index.date
    features["station_id"] = station_id
    features["parameter"] = target_param
    features["value"] = daily[target_param]

    # Time features
    features["month"] = daily["month"]
    features["day_of_week"] = daily["day_of_week"]
    features["is_weekend"] = daily["is_weekend"]
    features["day_of_year"] = daily["day_of_year"]

    # Lag features — short + long range
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        col = f"lag_{lag}"
        features[col] = daily[col] if col in daily.columns else None

    # Rolling features — 3d, 7d, 14d, 30d
    features["roll_3_mean"]  = daily.get("roll_3_mean")
    features["roll_7_mean"]  = daily.get("roll_7_mean")
    features["roll_3_std"]   = daily.get("roll_3_std")
    features["roll_14_mean"] = daily.get("roll_14_mean")
    features["roll_30_mean"] = daily.get("roll_30_mean")
    features["roll_14_std"]  = daily.get("roll_14_std")

    # Momentum / delta features
    features["pm25_delta_1"] = daily.get("pm25_delta_1")
    features["pm25_delta_7"] = daily.get("pm25_delta_7")

    # Cross-parameter features
    for feature_name, values in cross.items():
        features[feature_name] = values

    # Drop warmup rows — only require value and lag_1 (keep more rows)
    features = features.dropna(subset=["value", "lag_1"])

    return features


# ═══════════════════════════════════════════════════════════
# V2 BATCH DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════

def load_bulk_clean_data(conn, station_ids=None, lookback_days=90):
    """
    Load clean measurements for ALL target stations in ONE query.

    Uses lookback_days to limit data while ensuring enough history
    for lag/rolling features (max lag = 30 days, max rolling = 30 days,
    so 90 days provides ample buffer).

    Args:
        conn: psycopg2 connection
        station_ids: list of station IDs (None = all)
        lookback_days: load data from last N days (default 90)

    Returns:
        DataFrame with clean measurements
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        if station_ids is None:
            # Full rebuild: load everything
            return pd.read_sql(
                "SELECT station_id, parameter, value, datetime_local "
                "FROM clean_measurements WHERE is_valid = true "
                "ORDER BY station_id, datetime_local",
                conn
            )
        else:
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            return pd.read_sql(
                "SELECT station_id, parameter, value, datetime_local "
                "FROM clean_measurements "
                "WHERE station_id = ANY(%(ids)s) "
                "AND is_valid = true "
                "AND datetime_local >= %(cutoff)s "
                "ORDER BY station_id, datetime_local",
                conn,
                params={"ids": list(station_ids), "cutoff": cutoff}
            )


def bulk_insert_features(conn, features_df, page_size=5000):
    """
    Bulk upsert ALL feature rows in one batch operation.
    Uses execute_values with large page_size for maximum throughput.
    ON CONFLICT DO UPDATE ensures idempotent upserts.
    """
    if features_df.empty:
        return 0

    sql = """
        INSERT INTO daily_features
            (date, station_id, parameter, value,
             month, day_of_week, is_weekend, day_of_year,
             lag_1, lag_2, lag_3, lag_7,
             lag_14, lag_21, lag_30,
             roll_3_mean, roll_7_mean, roll_3_std,
             roll_14_mean, roll_30_mean, roll_14_std,
             pm25_delta_1, pm25_delta_7)
        VALUES %s
        ON CONFLICT (date, station_id, parameter) DO UPDATE SET
            value        = EXCLUDED.value,
            lag_1        = EXCLUDED.lag_1,
            lag_2        = EXCLUDED.lag_2,
            lag_3        = EXCLUDED.lag_3,
            lag_7        = EXCLUDED.lag_7,
            lag_14       = EXCLUDED.lag_14,
            lag_21       = EXCLUDED.lag_21,
            lag_30       = EXCLUDED.lag_30,
            roll_3_mean  = EXCLUDED.roll_3_mean,
            roll_7_mean  = EXCLUDED.roll_7_mean,
            roll_3_std   = EXCLUDED.roll_3_std,
            roll_14_mean = EXCLUDED.roll_14_mean,
            roll_30_mean = EXCLUDED.roll_30_mean,
            roll_14_std  = EXCLUDED.roll_14_std,
            pm25_delta_1 = EXCLUDED.pm25_delta_1,
            pm25_delta_7 = EXCLUDED.pm25_delta_7
    """

    # Column order must match the VALUES %s in the SQL
    insert_cols = [
        'date', 'station_id', 'parameter', 'value',
        'month', 'day_of_week', 'is_weekend', 'day_of_year',
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'lag_14', 'lag_21', 'lag_30',
        'roll_3_mean', 'roll_7_mean', 'roll_3_std',
        'roll_14_mean', 'roll_30_mean', 'roll_14_std',
        'pm25_delta_1', 'pm25_delta_7',
    ]

    values = []
    for row in features_df[insert_cols].itertuples(index=False):
        cleaned = []
        for i, val in enumerate(row):
            col_name = insert_cols[i]
            if pd.isna(val):
                cleaned.append(None)
            elif col_name == 'is_weekend':
                cleaned.append(bool(val))
            else:
                cleaned.append(val)
        values.append(tuple(cleaned))

    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=page_size)
    conn.commit()
    return len(values)


# ═══════════════════════════════════════════════════════════
# LEGACY OPERATIONS (backward compatibility)
# ═══════════════════════════════════════════════════════════

def build_features_for_station(conn, station_id, target_param="pm25"):
    """
    Legacy: Full feature pipeline for one station (loads from DB).
    Kept for backward compatibility with scripts that call it directly.
    For batch processing, use build_features_from_data() instead.
    """
    df = load_station_clean_data(conn, station_id)
    if df.empty:
        return None
    return build_features_from_data(df, station_id, target_param)


def insert_features(conn, features_df):
    """Legacy: Insert feature rows (delegates to bulk_insert_features)."""
    return bulk_insert_features(conn, features_df)


# ─── Database Write ───────────────────────────────────────
def ensure_v6_columns(conn):
    """
    Add v6 extended feature columns to daily_features if they don't exist.
    Safe to call multiple times — uses ADD COLUMN IF NOT EXISTS.
    """
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE daily_features
                ADD COLUMN IF NOT EXISTS lag_14       DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS lag_21       DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS lag_30       DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS roll_14_mean DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS roll_30_mean DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS roll_14_std  DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS pm25_delta_1 DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS pm25_delta_7 DOUBLE PRECISION
        """)
    conn.commit()


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def run_feature_pipeline(conn, station_ids=None, target_params=None,
                         lookback_days=90, insert_recent_days=7):
    """
    V2 Batch Feature Pipeline.

    Architecture:
        ONE bulk SQL query → in-memory groupby → ONE bulk upsert.
        The in-memory groupby still iterates per-station for lag/rolling
        computation, but ALL iterations are in-memory with ZERO network calls.

    Insert Window Safety:
        To avoid corrupting older features (which were computed from a fuller
        history), only features for the last `insert_recent_days` are upserted.
        The older lookback data is used purely for lag/rolling context.

    Args:
        conn: psycopg2 connection
        station_ids: list of station IDs (None = all)
        target_params: list of target parameters (default: ['pm25', 'pm10'])
        lookback_days: how many days of clean data to load for context (default 90)
        insert_recent_days: only upsert features within this window (default 7)
    """
    if target_params is None:
        target_params = ["pm25", "pm10"]

    # Ensure v6 columns exist before writing
    ensure_v6_columns(conn)

    if station_ids is None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT station_id FROM clean_measurements
                ORDER BY station_id
            """)
            station_ids = [row[0] for row in cur.fetchall()]

    # ── 1. SINGLE BULK LOAD ──
    print("  📦 Loading clean data (single bulk query)...")
    all_data = load_bulk_clean_data(conn, station_ids, lookback_days=lookback_days)

    if all_data.empty:
        print("  ⚠️  No clean data found.")
        return {"stations_with_features": 0, "total_features": 0}

    total_stations = all_data['station_id'].nunique()
    print(f"  Loaded {len(all_data):,} rows across {total_stations} stations")

    # ── 2. IN-MEMORY FEATURE ENGINEERING (groupby, zero DB calls) ──
    print("  🛠️  Computing features (in-memory groupby)...")
    all_features = []
    grouped = all_data.groupby('station_id')
    stations_with_data = 0

    for i, (sid, group) in enumerate(grouped):
        station_features_count = 0
        for param in target_params:
            features = build_features_from_data(group, sid, param)
            if features is not None and not features.empty:
                all_features.append(features)
                station_features_count += len(features)

        if station_features_count > 0:
            stations_with_data += 1

        # Progress every 500 stations
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{total_stations}] stations processed...")

    if not all_features:
        print("  ⚠️  No features generated.")
        return {"stations_with_features": 0, "total_features": 0}

    combined = pd.concat(all_features, ignore_index=True)
    print(f"  Generated {len(combined):,} feature rows from {stations_with_data} stations")

    # ── 2b. INSERT WINDOW FILTER ──
    # Only upsert features for the last N days to avoid corrupting
    # older features that were computed from a fuller history.
    if insert_recent_days and station_ids is not None:
        cutoff_date = (datetime.utcnow() - timedelta(days=insert_recent_days)).date()
        before_count = len(combined)
        combined = combined[combined['date'] >= cutoff_date]
        print(f"  🔒 Insert window: last {insert_recent_days} days → "
              f"{before_count:,} → {len(combined):,} rows")

    if combined.empty:
        print("  ⚠️  No features within insert window.")
        return {"stations_with_features": stations_with_data, "total_features": 0}

    # ── 3. SINGLE BULK INSERT ──
    print(f"  💾 Bulk upserting {len(combined):,} feature rows...")
    inserted = bulk_insert_features(conn, combined)

    print(f"\n  ✅ Feature engineering complete:")
    print(f"     Stations with features: {stations_with_data}")
    print(f"     Total feature rows upserted: {len(combined):,}")

    return {
        "stations_with_features": stations_with_data,
        "total_features": len(combined),
    }


# ═══════════════════════════════════════════════════════════
# ADVANCED WEATHER FEATURES (already efficient — SQL window)
# ═══════════════════════════════════════════════════════════

def ensure_advanced_weather_columns(conn):
    """Ensure the advanced engineered weather/AOD columns exist."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE daily_features
                ADD COLUMN IF NOT EXISTS rolling_3day_precip DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS aod_volatility_index DOUBLE PRECISION
        """)
    conn.commit()


def build_advanced_weather_features(conn):
    """
    Computes advanced weather features like 3-day rolling precipitation
    and AOD volatility across all rows using efficient Postgres window functions.
    This prevents data leakage by explicitly skipping the current day.
    """
    ensure_advanced_weather_columns(conn)

    # We update all rows that don't have rolling_3day_precip computed yet.
    # To compute rolling features safely without data leakage:
    # rolling_3day_precip: SUM of past 3 days (excluding today).
    # aod_volatility: STDDEV of past 7 days (excluding today).

    sql = """
        WITH rolling_data AS (
            SELECT station_id, date, parameter,
                   SUM(om_precipitation) OVER (
                       PARTITION BY station_id, parameter
                       ORDER BY date
                       ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                   ) as roll_3_precip,
                   STDDEV(om_aerosol_optical_depth) OVER (
                       PARTITION BY station_id, parameter
                       ORDER BY date
                       ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                   ) as aod_vol
            FROM daily_features
        )
        UPDATE daily_features df
        SET rolling_3day_precip = COALESCE(rd.roll_3_precip, 0),
            aod_volatility_index = COALESCE(rd.aod_vol, 0)
        FROM rolling_data rd
        WHERE df.station_id = rd.station_id
          AND df.date = rd.date
          AND df.parameter = rd.parameter
          AND (df.rolling_3day_precip IS NULL OR df.aod_volatility_index IS NULL)
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        updated = cur.rowcount
    conn.commit()
    return updated
