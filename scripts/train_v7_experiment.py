"""
Global Air Quality Model — v6 Direct Multi-Horizon Training
============================================================
Per-country, per-horizon GradientBoostingRegressor.

Problem with v5 chaining:
  Day-1 prediction feeds into Day-2 as lag_1, which feeds into Day-3, etc.
  Each step compounds the error — by Day-30 the model is predicting noise.

v6 fix — direct horizon models:
  Train one separate model per target horizon (1d, 7d, 14d, 30d).
  Each model sees real observed lag features and predicts DIRECTLY
  to that horizon. No error propagation. No chaining.

v6 also adds richer features (via features.py v6 pipeline):
  - lag_14, lag_21, lag_30  → monthly pollution memory
  - roll_14_mean/std        → 2-week trend context
  - roll_30_mean            → monthly baseline
  - pm25_delta_1/7          → momentum (is pollution rising or falling?)

Usage:
    python scripts/train_v6.py
    python scripts/train_v6.py --countries IN US  # specific countries
"""

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

# ─── Config ───────────────────────────────────────────────
HORIZONS = [1, 7, 14, 30]
COUNTRIES = ["IN", "US", "GB", "AU"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v7")  # Production

# v6 feature set — all backward-looking, zero leakage
FEATURE_COLS_V6 = [
    # Today's actual value (lag_0)
    "value",
    # Calendar
    "month", "day_of_week", "is_weekend", "day_of_year",
    # Short-range lags (recent memory)
    "lag_1", "lag_2", "lag_3", "lag_7",
    # Long-range lags (monthly context — new in v6)
    "lag_14", "lag_21", "lag_30",
    # Rolling means
    "roll_3_mean", "roll_7_mean",
    "roll_14_mean", "roll_30_mean",
    # Rolling std (volatility)
    "roll_3_std", "roll_14_std",
    # Momentum / first-difference (new in v6)
    "pm25_delta_1", "pm25_delta_7",
    # In-situ weather
    "temperature", "humidity", "wind_speed",
    "no2_value", "co_value", "o3_value", "so2_value",
    # NASA POWER satellite weather
    "nasa_temperature", "nasa_humidity", "nasa_wind_speed",
    "precipitation", "wind_direction",
    # Fire
    "fire_count",
]


# ─── Data Loading ─────────────────────────────────────────
def load_country_features(conn, country_code):
    """Load PM2.5 daily features for one country."""
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


def get_available_features(df):
    """Return v7 features that have >10% non-null coverage."""
    available = []
    n = len(df)
    for col in FEATURE_COLS_V6 + ["future_temp", "future_wind", "future_precip"]:
        if col in df.columns:
            if df[col].notna().sum() / n > 0.10:
                available.append(col)
    return available


# ─── Target Engineering ───────────────────────────────────
def make_horizon_target(df, horizon):
    """
    For each station, look exactly `horizon` calendar days ahead to create
    the direct-horizon target and pull future weather features.
    """
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    
    # Grab target PM2.5 and Open-Meteo target weather
    cols = ["station_id", "date", "value", "om_temperature", "om_wind_speed", "om_precipitation"]
    
    # Ensure columns exist (in case some countries lack them)
    cols = [c for c in cols if c in df_copy.columns]
    
    target_df = df_copy[cols].copy()
    target_df = target_df.rename(columns={
        "date": "target_date", 
        "value": f"target_h{horizon}",
        "om_temperature": "future_temp",
        "om_wind_speed": "future_wind",
        "om_precipitation": "future_precip"
    })
    
    df_copy["target_date"] = df_copy["date"] + pd.Timedelta(days=horizon)
    
    # Drop om_ columns from df_copy to prevent duplicates after merge
    # (the future_ versions from target_df are the ones we want)
    om_cols_to_drop = [c for c in ["om_temperature", "om_wind_speed", "om_precipitation"] if c in df_copy.columns]
    df_copy = df_copy.drop(columns=om_cols_to_drop)
    
    result = pd.merge(df_copy, target_df, on=["station_id", "target_date"], how="inner")
    return result


# ─── Temporal Split ───────────────────────────────────────
def temporal_split(df, target_col, test_ratio=0.20):
    """
    Strict per-station temporal split.
    Train = first 80% of each station's timeline.
    Test  = last 20%.  No shuffling, no station overlap.
    Enforces target_date < split_date for train to prevent boundary leakage.
    """
    train_parts, test_parts = [], []

    for sid, group in df.groupby("station_id"):
        group = group.sort_values("date")
        n = len(group)
        split_idx = int(n * (1 - test_ratio))

        if split_idx < 10 or (n - split_idx) < 3:
            continue

        split_date = group.iloc[split_idx]["date"]
        
        train_mask = group["target_date"] < split_date
        test_mask  = group["date"] >= split_date
        
        train_parts.append(group[train_mask])
        test_parts.append(group[test_mask])

    if not train_parts:
        return pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


# ─── Per-Horizon Training ─────────────────────────────────
def train_horizon_model(conn, country_code, horizon):
    """Train one direct model: country × horizon."""
    print(f"\n  ── {country_code}  h={horizon:>2}d ──")

    df = load_country_features(conn, country_code)
    if len(df) < 100:
        print(f"     ⚠️  Only {len(df)} rows — skipping")
        return None

    # Build horizon target
    df_h = make_horizon_target(df, horizon)
    target_col = f"target_h{horizon}"
    if len(df_h) < 50:
        print(f"     ⚠️  Only {len(df_h)} rows with h{horizon} target — skipping")
        return None

    # Available features
    features = get_available_features(df_h)
    core = ["lag_1", "lag_2", "lag_3", "month", "day_of_week"]
    missing_core = [c for c in core if c not in features]
    if missing_core:
        print(f"     ⚠️  Missing core features: {missing_core}")
        return None

    # Temporal split
    train, test = temporal_split(df_h, target_col)
    if train.empty or test.empty:
        print(f"     ⚠️  Split failed")
        return None

    print(f"     rows  train={len(train):,}  test={len(test):,}  feats={len(features)}")
    print(f"     dates train={train['date'].min().date()}→{train['date'].max().date()}")
    print(f"           test ={test['date'].min().date()}→{test['date'].max().date()}")

    # Prepare arrays
    X_train = train[features].copy()
    y_train = train[target_col].copy()
    X_test  = test[features].copy()
    y_test  = test[target_col].copy()

    # Fill NaNs with training medians (robust to sparse weather columns)
    medians = {}
    for col in features:
        med = X_train[col].median()
        if hasattr(med, '__len__'):
            med = med.iloc[0] if len(med) > 0 else 0.0
        medians[col] = med if not (isinstance(med, float) and pd.isna(med)) else 0.0
        X_train[col] = X_train[col].fillna(medians[col])
        X_test[col]  = X_test[col].fillna(medians[col])

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test  = X_test.replace([np.inf, -np.inf], 0)

    # GBR — same hyperparams as v5 for fair comparison
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Metrics
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_r2  = r2_score(y_train, y_pred_train)
    test_r2   = r2_score(y_test, y_pred_test)
    test_mae  = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"     train R²={train_r2:.4f}  test R²={test_r2:.4f}  "
          f"MAE={test_mae:.2f}  RMSE={test_rmse:.2f}  "
          f"Δ={train_r2-test_r2:+.4f}")

    # Feature importance
    importances = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"     top-3: {', '.join(f'{n}({v:.3f})' for n,v in importances[:3])}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_h{horizon}_gbr.pkl")
    joblib.dump(model, model_path)

    meta = {
        "country":        country_code,
        "model":          "GradientBoostingRegressor",
        "version":        "v7_weather_direct",
        "horizon_days":   horizon,
        "features":       features,
        "feature_medians": medians,
        "train_rows":     len(train),
        "test_rows":      len(test),
        "train_stations": int(train["station_id"].nunique()),
        "train_date_range": [
            str(train["date"].min().date()),
            str(train["date"].max().date()),
        ],
        "test_date_range": [
            str(test["date"].min().date()),
            str(test["date"].max().date()),
        ],
        "metrics": {
            "train_r2":      round(train_r2,  4),
            "test_r2":       round(test_r2,   4),
            "test_mae":      round(test_mae,  2),
            "test_rmse":     round(test_rmse, 2),
            "overfit_delta": round(train_r2 - test_r2, 4),
        },
        "top_features": [
            (f, round(float(i), 4)) for f, i in importances[:10]
        ],
    }
    meta_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_h{horizon}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ─── Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--countries", nargs="+", default=COUNTRIES,
        help="Countries to train (default: all)"
    )
    args = parser.parse_args()

    start = time.time()
    conn  = psycopg2.connect(**DB_CONFIG)

    print("═" * 62)
    print("  GLOBAL AQ MODEL v6 — DIRECT MULTI-HORIZON TRAINING")
    print("═" * 62)
    print(f"  Strategy : one model per horizon, no error chaining")
    print(f"  Horizons : {HORIZONS}")
    print(f"  Countries: {args.countries}")
    print(f"  New feats: lag_14/21/30  roll_14/30  delta_1/7")
    print()

    all_meta = {}

    for cc in args.countries:
        print(f"\n{'═'*62}")
        print(f"  {cc}")
        print(f"{'─'*62}")
        all_meta[cc] = {}

        for h in HORIZONS:
            meta = train_horizon_model(conn, cc, h)
            if meta:
                all_meta[cc][f"h{h}"] = meta

    # ── Summary table ─────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'═'*62}")
    print(f"  COMPLETE  ({int(elapsed)}s)")
    print(f"{'═'*62}")
    print(f"\n  {'CC':>3}  {'H':>4}  {'Train R²':>10}  {'Test R²':>10}  "
          f"{'MAE':>8}  {'RMSE':>8}  {'Overfit':>9}")
    print(f"  {'─'*58}")

    for cc in args.countries:
        for h in HORIZONS:
            key = f"h{h}"
            if cc in all_meta and key in all_meta[cc]:
                m = all_meta[cc][key]["metrics"]
                print(
                    f"  {cc:>3}  {h:>3}d  "
                    f"{m['train_r2']:>10.4f}  {m['test_r2']:>10.4f}  "
                    f"{m['test_mae']:>8.2f}  {m['test_rmse']:>8.2f}  "
                    f"{m['overfit_delta']:>+9.4f}"
                )

    # Save combined metadata
    combined_path = os.path.join(MODEL_DIR, "all_models_meta.json")
    with open(combined_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n  Models  → models/v6/")
    print(f"  Metadata→ {combined_path}")
    conn.close()


if __name__ == "__main__":
    main()
