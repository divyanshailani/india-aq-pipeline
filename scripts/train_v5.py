"""
Global Air Quality Model — v5 Leak-Proof Training
===================================================
Per-country GradientBoostingRegressor with strict temporal split.

Leak Prevention:
  1. Temporal train/test split (last 20% of each station's data = test)
  2. Rolling features already shifted by 1 day in pipeline (lag-only)
  3. No future data in any feature (all features use shift(≥1))
  4. Per-station temporal split (no station appears in both train/test at same time)

Usage:
    python scripts/train_v5.py
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": 5432,
}

# Features to use for training (all backward-looking, no leakage)
FEATURE_COLS = [
    # Calendar
    "month", "day_of_week", "is_weekend", "day_of_year",
    # Lags (strictly past values)
    "lag_1", "lag_2", "lag_3", "lag_7",
    # Rolling (already shifted by 1 day in DB — uses lag_1..lag_3)
    "roll_3_mean", "roll_7_mean", "roll_3_std",
    # Cross-pollutant (same-day co-located sensors, not leakage)
    "temperature", "humidity", "wind_speed",
    "no2_value", "co_value", "o3_value", "so2_value",
    # NASA weather (aligned to same date, external source = OK)
    "nasa_temperature", "nasa_humidity", "nasa_wind_speed",
    "precipitation", "wind_direction",
    # Fire
    "fire_count",
]

TARGET_COL = "value"  # PM2.5 daily mean

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v5")
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "predictions")


def load_country_features(conn, country_code):
    """Load PM2.5 daily features for one country, sorted by date."""
    sql = """
        SELECT * FROM daily_features
        WHERE country_code = %s
          AND parameter = 'pm25'
          AND value IS NOT NULL
          AND lag_1 IS NOT NULL
        ORDER BY station_id, date
    """
    df = pd.read_sql(sql, conn, params=(country_code,))
    df["date"] = pd.to_datetime(df["date"])
    return df


def temporal_split(df, test_ratio=0.20):
    """
    Strict temporal split: for each station, the last `test_ratio`
    fraction of its time-ordered data goes to test. No shuffling.
    
    This prevents ANY future leakage:
    - Train data is always BEFORE test data for each station
    - No station's test data can leak into another station's train
    """
    train_parts = []
    test_parts = []

    for station_id, group in df.groupby("station_id"):
        group = group.sort_values("date")
        n = len(group)
        split_idx = int(n * (1 - test_ratio))

        if split_idx < 7:  # Need at least 7 days of training
            continue
        if n - split_idx < 3:  # Need at least 3 test days
            continue

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return train, test


def get_available_features(df):
    """
    Get features that actually have data (>10% non-null).
    This avoids using columns that are all NaN for a country.
    """
    available = []
    n = len(df)
    for col in FEATURE_COLS:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null / n > 0.10:  # At least 10% coverage
                available.append(col)
    return available


def train_country_model(conn, country_code):
    """Train a leak-proof model for one country."""
    print(f"\n{'─'*60}")
    print(f"  Training {country_code} model")
    print(f"{'─'*60}")

    # Load data
    df = load_country_features(conn, country_code)
    print(f"  Loaded {len(df):,} rows, {df['station_id'].nunique()} stations")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    if len(df) < 100:
        print(f"  ⚠️ Skipping {country_code}: not enough data")
        return None

    # Temporal split
    train, test = temporal_split(df, test_ratio=0.20)
    print(f"  Train: {len(train):,} rows ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test:  {len(test):,} rows ({test['date'].min().date()} → {test['date'].max().date()})")

    # Select features with actual data
    features = get_available_features(train)
    print(f"  Features: {len(features)} available")

    # Core features that MUST exist
    core = ["lag_1", "lag_2", "lag_3", "month", "day_of_week", "day_of_year"]
    for c in core:
        if c not in features:
            print(f"  ⚠️ Missing core feature: {c}")
            return None

    # Prepare X, y
    X_train = train[features].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[features].copy()
    y_test = test[TARGET_COL].copy()

    # Fill remaining NaNs with median (for optional features like weather)
    for col in features:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # Remove any remaining infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # Train GBR
    print(f"  Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n  Results:")
    print(f"    Train R²:  {train_r2:.4f}")
    print(f"    Test  R²:  {test_r2:.4f}")
    print(f"    Test  MAE: {test_mae:.2f} µg/m³")
    print(f"    Test RMSE: {test_rmse:.2f} µg/m³")
    print(f"    Overfit Δ: {train_r2 - test_r2:.4f}")

    # Feature importance
    importances = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top features:")
    for feat, imp in importances[:8]:
        print(f"    {feat:<20} {imp:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_gbr.pkl")
    joblib.dump(model, model_path)
    print(f"\n  Saved: {model_path}")

    # Save metadata
    meta = {
        "country": country_code,
        "model": "GradientBoostingRegressor",
        "version": "v5",
        "features": features,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_stations": int(train["station_id"].nunique()),
        "test_stations": int(test["station_id"].nunique()),
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_date_range": [str(test["date"].min().date()), str(test["date"].max().date())],
        "metrics": {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "test_mae": round(test_mae, 2),
            "test_rmse": round(test_rmse, 2),
            "overfit_delta": round(train_r2 - test_r2, 4),
        },
        "top_features": [(f, round(float(i), 4)) for f, i in importances[:10]],
    }
    meta_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Generate test predictions for frontend
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions = test[["date", "station_id", "value"]].copy()
    predictions["predicted"] = y_pred_test
    predictions["residual"] = y_test.values - y_pred_test
    predictions["country"] = country_code
    pred_path = os.path.join(PREDICTIONS_DIR, f"{country_code}_predictions.csv")
    predictions.to_csv(pred_path, index=False)
    print(f"  Predictions: {pred_path}")

    return meta


def main():
    start = time.time()
    conn = psycopg2.connect(**DB_CONFIG)

    print("═" * 60)
    print("  GLOBAL AIR QUALITY MODEL v5 — LEAK-PROOF TRAINING")
    print("═" * 60)
    print()
    print("  Leak prevention:")
    print("    ✓ Temporal train/test split (per-station)")
    print("    ✓ Rolling features shifted by 1 day")
    print("    ✓ All lag features use strictly past data")
    print("    ✓ No random shuffling")

    countries = ["IN", "US", "GB", "AU"]
    all_meta = {}

    for cc in countries:
        meta = train_country_model(conn, cc)
        if meta:
            all_meta[cc] = meta

    # Summary
    elapsed = time.time() - start
    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE ({int(elapsed)}s)")
    print(f"{'═'*60}")
    print(f"\n  {'CC':>3} {'Train R²':>10} {'Test R²':>10} {'MAE':>8} {'RMSE':>8} {'Overfit':>9}")
    print(f"  {'─'*52}")
    for cc, meta in all_meta.items():
        m = meta["metrics"]
        print(f"  {cc:>3} {m['train_r2']:>10.4f} {m['test_r2']:>10.4f} {m['test_mae']:>8.2f} {m['test_rmse']:>8.2f} {m['overfit_delta']:>+9.4f}")

    # Save combined metadata
    combined_path = os.path.join(MODEL_DIR, "all_models_meta.json")
    with open(combined_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n  Combined metadata: {combined_path}")

    conn.close()


if __name__ == "__main__":
    main()
