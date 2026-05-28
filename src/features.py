"""
IndiaAQ Intelligence — Feature Engineering Pipeline
====================================================
Reads from clean_measurements, engineers features,
writes to daily_features.

Features per station per day:
    - Time:    month, day_of_week, is_weekend, day_of_year
    - Lag:     lag_1, lag_2, lag_3, lag_7
    - Rolling: roll_3_mean, roll_7_mean, roll_3_std
    - Cross:   temperature, humidity, wind_speed, no2, co, o3, so2
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


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
def add_lag_features(daily, target_col, lags=[1, 2, 3, 7]):
    """
    Add lagged versions of the target column.
    lag_1 = yesterday's value, lag_7 = last week's value.
    """
    for lag in lags:
        daily[f"lag_{lag}"] = daily[target_col].shift(lag)
    return daily


# ─── Step 5: Add Rolling Features ─────────────────────────
def add_rolling_features(daily, target_col, windows=[3, 7]):
    """
    Add rolling mean and std for the target column.
    roll_3_mean = average of last 3 days.
    """
    for w in windows:
        daily[f"roll_{w}_mean"] = daily[target_col].rolling(w).mean()
    # Only add std for shortest window (volatility indicator)
    daily["roll_3_std"] = daily[target_col].rolling(3).std()
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


# ─── Full Feature Pipeline ────────────────────────────────
def build_features_for_station(conn, station_id, target_param="pm25"):
    """
    Full feature pipeline for one station.

    Args:
        conn: database connection
        station_id: internal station ID
        target_param: which parameter to predict ('pm25' or 'pm10')

    Returns:
        DataFrame with features, or None if insufficient data
    """
    # Load clean data
    df = load_station_clean_data(conn, station_id)
    if df.empty:
        return None

    # Need at least the target parameter
    if target_param not in df["parameter"].values:
        return None

    # Pivot to daily wide format
    daily = make_daily_wide(df)

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

    # Lag features
    for lag in [1, 2, 3, 7]:
        col = f"lag_{lag}"
        features[col] = daily[col] if col in daily.columns else None

    # Rolling features
    features["roll_3_mean"] = daily.get("roll_3_mean")
    features["roll_7_mean"] = daily.get("roll_7_mean")
    features["roll_3_std"] = daily.get("roll_3_std")

    # Cross-parameter features
    for feature_name, values in cross.items():
        features[feature_name] = values

    # Drop warmup rows (NaN from lag/rolling)
    features = features.dropna(subset=["value", "lag_7", "roll_7_mean"])

    return features


# ─── Database Write ───────────────────────────────────────
def insert_features(conn, features_df):
    """Insert feature rows into daily_features table."""
    if features_df.empty:
        return 0

    sql = """
        INSERT INTO daily_features
            (date, station_id, parameter, value,
             month, day_of_week, is_weekend, day_of_year,
             lag_1, lag_2, lag_3, lag_7,
             roll_3_mean, roll_7_mean, roll_3_std,
             temperature, humidity, wind_speed,
             no2_value, co_value, o3_value, so2_value)
        VALUES %s
        ON CONFLICT (date, station_id, parameter) DO UPDATE SET
            value = EXCLUDED.value,
            lag_1 = EXCLUDED.lag_1,
            lag_2 = EXCLUDED.lag_2,
            lag_3 = EXCLUDED.lag_3,
            lag_7 = EXCLUDED.lag_7,
            roll_3_mean = EXCLUDED.roll_3_mean,
            roll_7_mean = EXCLUDED.roll_7_mean,
            roll_3_std = EXCLUDED.roll_3_std,
            temperature = EXCLUDED.temperature,
            humidity = EXCLUDED.humidity,
            wind_speed = EXCLUDED.wind_speed
    """

    values = []
    for _, row in features_df.iterrows():
        values.append((
            row["date"], row["station_id"], row["parameter"], row.get("value"),
            row.get("month"), row.get("day_of_week"),
            bool(row.get("is_weekend")) if pd.notna(row.get("is_weekend")) else None,
            row.get("day_of_year"),
            row.get("lag_1"), row.get("lag_2"), row.get("lag_3"), row.get("lag_7"),
            row.get("roll_3_mean"), row.get("roll_7_mean"), row.get("roll_3_std"),
            row.get("temperature"), row.get("humidity"), row.get("wind_speed"),
            row.get("no2_value"), row.get("co_value"),
            row.get("o3_value"), row.get("so2_value"),
        ))

    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    return len(values)


# ─── Main Entry Point ────────────────────────────────────
def run_feature_pipeline(conn, station_ids=None, target_params=None):
    """
    Run feature engineering on all (or specified) stations.

    Args:
        conn: psycopg2 connection
        station_ids: list of station IDs (None = all with clean data)
        target_params: list of target parameters (default: ['pm25', 'pm10'])
    """
    if target_params is None:
        target_params = ["pm25", "pm10"]

    if station_ids is None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT station_id FROM clean_measurements
                ORDER BY station_id
            """)
            station_ids = [row[0] for row in cur.fetchall()]

    total_stations = len(station_ids)
    total_features = 0
    stations_with_data = 0

    for i, sid in enumerate(station_ids):
        station_total = 0

        for param in target_params:
            features = build_features_for_station(conn, sid, param)
            if features is not None and not features.empty:
                inserted = insert_features(conn, features)
                station_total += inserted

        if station_total > 0:
            stations_with_data += 1
            total_features += station_total
            print(
                f"  [{i+1}/{total_stations}] Station {sid}: "
                f"{station_total} feature rows"
            )

    print(f"\n  ✅ Feature engineering complete:")
    print(f"     Stations with features: {stations_with_data}")
    print(f"     Total feature rows: {total_features:,}")

    return {
        "stations_with_features": stations_with_data,
        "total_features": total_features,
    }
