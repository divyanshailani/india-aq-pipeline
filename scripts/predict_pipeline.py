"""
Global AQ Intelligence — Prediction Pipeline
==============================================
One-click pipeline: fetch → validate → predict → export JSON.

Horizons:
  - 7-day:  Direct model prediction (high confidence)
  - 15-day: Chained predictions with widening confidence bands
  - 30-day: Chained predictions with "decreasing accuracy" warning

Usage:
    python scripts/predict_pipeline.py              # full run
    python scripts/predict_pipeline.py --skip-fetch  # skip data fetch, just predict
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, date

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG, MODEL_DIR as _MODEL_DIR, SITE_DATA_DIR

MODEL_DIR = _MODEL_DIR
OUTPUT_DIR = SITE_DATA_DIR

COUNTRIES = ["IN", "US", "GB", "AU"]
ACTIVE_STATION_MAX_AGE_DAYS = 7
RECENT_CONTEXT_DAYS = 14
MIN_LIVE_VALIDATIONS = 100

COUNTRY_META = {
    "IN": {
        "name": "India",
        "flag": "🇮🇳",
        "confidence": "high",
        "tag": "High Confidence",
        "tag_color": "green",
        "reason": "R²=0.64 on 31K features, CPCB reference + NASA weather/fire data",
        "test_r2": 0.6399,
        "test_mae": 10.18,
    },
    "US": {
        "name": "United States",
        "flag": "🇺🇸",
        "confidence": "high",
        "tag": "High Confidence",
        "tag_color": "green",
        "reason": "R²=0.68 on 1.4M features, EPA AQS reference-grade stations",
        "test_r2": 0.6783,
        "test_mae": 2.01,
    },
    "GB": {
        "name": "United Kingdom",
        "flag": "🇬🇧",
        "confidence": "experimental",
        "tag": "Experimental: Limited Seasonal Data",
        "tag_color": "yellow",
        "reason": "R²=0.21, fragmented community sensors + limited data",
        "test_r2": 0.2082,
        "test_mae": 2.68,
    },
    "AU": {
        "name": "Australia",
        "flag": "🇦🇺",
        "confidence": "stable",
        "tag": "Low Variance / Stable",
        "tag_color": "blue",
        "reason": "R²=0.60, clean-air country with NSW EPA reference data",
        "test_r2": 0.6017,
        "test_mae": 1.61,
    },
}


def get_last_run_date(conn):
    """Get the date of the last pipeline run."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(run_date) FROM pipeline_runs WHERE status = 'completed'")
        result = cur.fetchone()[0]
    return result.date() if result else None


def ensure_tracking_schema(conn):
    """Keep local tracking tables compatible with current metrics reporting."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE pipeline_runs
                ADD COLUMN IF NOT EXISTS backtest_mae DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS backtest_r2 DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS metric_source TEXT,
                ADD COLUMN IF NOT EXISTS metric_sample_count INTEGER
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_log_run_target
                ON prediction_log (run_date, target_date, country_code)
        """)
    conn.commit()


def validate_old_predictions(conn, run_id):
    """
    Compare old predictions (where target_date <= today and actual is NULL)
    against real data that has arrived.
    """
    today = date.today()
    validated = 0

    with conn.cursor() as cur:
        # Only validate plausible completed forecasts.
        # target_date >= run_date quarantines old bad rows generated from stale
        # station anchors (for example, a 2026 run forecasting from 2021 data).
        cur.execute("""
            SELECT pl.id, pl.station_id, pl.target_date, pl.predicted_value, pl.country_code
            FROM prediction_log pl
            WHERE pl.actual_value IS NULL
              AND pl.target_date < %s
              AND pl.target_date >= pl.run_date
        """, (today,))
        pending = cur.fetchall()

        if pending:
            print(f"  Found {len(pending)} predictions to validate...")
        else:
            print("  No predictions to validate")

        for pid, station_id, target_date, predicted, cc in pending:
            # Validate against daily_features, the same daily target table used
            # for training and forecasting.
            cur.execute("""
                SELECT value FROM daily_features
                WHERE station_id = %s
                  AND parameter = 'pm25'
                  AND date = %s
                  AND value IS NOT NULL
                LIMIT 1
            """, (station_id, target_date))
            result = cur.fetchone()
            actual = result[0] if result and result[0] is not None else None

            if actual is not None:
                error = actual - predicted
                cur.execute("""
                    UPDATE prediction_log
                    SET actual_value = %s, error = %s, validated_at = NOW()
                    WHERE id = %s
                """, (actual, error, pid))
                validated += 1

        conn.commit()

        # Calculate live metrics only when the sample is large enough to trust.
        cur.execute("""
            SELECT actual_value, predicted_value
            FROM prediction_log
            WHERE actual_value IS NOT NULL
              AND validated_at >= NOW() - INTERVAL '90 days'
              AND target_date >= run_date
        """)
        rows = cur.fetchall()

        live_sample_count = len(rows)
        if live_sample_count >= MIN_LIVE_VALIDATIONS:
            actuals = np.array([r[0] for r in rows])
            preds = np.array([r[1] for r in rows])
            live_mae = float(np.mean(np.abs(actuals - preds)))
            ss_res = np.sum((actuals - preds) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            live_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            live_mae = None
            live_r2 = None

    print(f"  Validated: {validated}/{len(pending)}")
    if live_mae is not None:
        print(f"  Live MAE:  {live_mae:.2f} µg/m³")
        print(f"  Live R²:   {live_r2:.4f}")
    else:
        print(f"  Live metrics hidden until {MIN_LIVE_VALIDATIONS}+ validations")

    return validated, live_mae, live_r2, live_sample_count


def backtest_recent(conn, n_days=7):
    """
    Backtest: predict the last N days of known data and compare to actuals.
    Gives fresh accuracy metrics every run without waiting for future validation.
    """
    print(f"\n  Backtesting last {n_days} days against actuals...")
    
    all_actuals = []
    all_preds = []
    country_metrics = {}

    for cc in COUNTRIES:
        meta_path = os.path.join(MODEL_DIR, f"{cc}_pm25_meta.json")
        model_path = os.path.join(MODEL_DIR, f"{cc}_pm25_gbr.pkl")
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta["features"]

        # Recent one-step backtest: known actual rows, using already-built lag features.
        sql = """
            SELECT * FROM daily_features
            WHERE country_code = %s
              AND parameter = 'pm25'
              AND value IS NOT NULL
              AND lag_1 IS NOT NULL
              AND date >= CURRENT_DATE - (%s * INTERVAL '1 day')
            ORDER BY station_id, date
        """
        df = pd.read_sql(sql, conn, params=(cc, n_days + RECENT_CONTEXT_DAYS))
        if df.empty or len(df) < 10:
            continue

        # Only predict rows from last N days (but use older rows for context)
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
        test_mask = pd.to_datetime(df["date"]) >= cutoff

        # Ensure feature columns exist
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            continue

        test_df = df[test_mask].copy()
        if test_df.empty:
            continue

        # Fill missing features with 0 and preserve training feature order.
        for col in feature_cols:
            if col not in test_df.columns:
                test_df[col] = 0
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0)

        X_test = test_df[feature_cols]
        y_actual = test_df["value"].values

        try:
            y_pred = model.predict(X_test)
        except Exception:
            continue

        mae = float(np.mean(np.abs(y_actual - y_pred)))
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        country_metrics[cc] = {"r2": round(r2, 4), "mae": round(mae, 2), "n": len(y_actual)}
        all_actuals.extend(y_actual.tolist())
        all_preds.extend(y_pred.tolist())

        print(f"    {cc}: R²={r2:.4f}  MAE={mae:.2f} µg/m³  ({len(y_actual)} samples)")

    if all_actuals:
        actuals = np.array(all_actuals)
        preds = np.array(all_preds)
        overall_mae = float(np.mean(np.abs(actuals - preds)))
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        overall_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"\n  📊 Backtest (last {n_days}d):")
        print(f"     Overall R²:  {overall_r2:.4f}")
        print(f"     Overall MAE: {overall_mae:.2f} µg/m³")
        print(f"     Samples:     {len(actuals):,}")

        return overall_mae, overall_r2, len(actuals), country_metrics
    
    return None, None, 0, {}


def get_recent_features(conn, country_code, n_days=RECENT_CONTEXT_DAYS,
                        active_days=ACTIVE_STATION_MAX_AGE_DAYS):
    """Get recent context only for stations that are still active."""
    sql = """
        WITH active_stations AS (
            SELECT station_id, MAX(date) AS last_date, COUNT(*) AS total_rows
            FROM daily_features
            WHERE country_code = %s
              AND parameter = 'pm25'
              AND value IS NOT NULL
              AND lag_1 IS NOT NULL
            GROUP BY station_id
            HAVING MAX(date) >= CURRENT_DATE - (%s * INTERVAL '1 day')
        ),
        ranked AS (
            SELECT df.*, a.last_date, a.total_rows,
                   ROW_NUMBER() OVER (PARTITION BY df.station_id ORDER BY df.date DESC) as rn
            FROM daily_features df
            JOIN active_stations a ON a.station_id = df.station_id
            WHERE df.country_code = %s
              AND df.parameter = 'pm25'
              AND df.value IS NOT NULL
              AND df.lag_1 IS NOT NULL
        )
        SELECT * FROM ranked WHERE rn <= %s
        ORDER BY station_id, date
    """
    return pd.read_sql(sql, conn, params=(country_code, active_days, country_code, n_days))


def predict_horizon(model, features, last_row, horizon_days, meta_path):
    """
    Generate predictions for 1..horizon_days.
    
    ⚠️  EXPERIMENTAL: 30-day chained forecast.
    Confidence degrades as predictions feed back as inputs:
      - Days 1-7:  HIGH   — direct prediction from real observed lags
      - Days 8-15: MEDIUM — chained from earlier predictions
      - Days 16-30: LOW   — deeply chained, treat as trend indication only
    
    This is useful for demo/trend visualization but should NOT be used
    for operational decisions beyond day 7 without validation.
    """
    # Load feature list from metadata
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    predictions = []
    current_lags = {
        "lag_1": last_row.get("value", last_row.get("lag_1", 0)),
        "lag_2": last_row.get("lag_1", 0),
        "lag_3": last_row.get("lag_2", 0),
        "lag_7": last_row.get("lag_3", 0),  # approximate
    }
    roll_values = [current_lags["lag_1"], current_lags["lag_2"], current_lags["lag_3"]]

    last_date = pd.to_datetime(last_row.get("date", date.today()))

    for day in range(1, horizon_days + 1):
        target_date = last_date + timedelta(days=day)

        # Build feature vector
        row = {}
        row["month"] = target_date.month
        row["day_of_week"] = target_date.weekday()
        row["is_weekend"] = 1 if target_date.weekday() >= 5 else 0
        row["day_of_year"] = target_date.timetuple().tm_yday
        row["lag_1"] = current_lags["lag_1"]
        row["lag_2"] = current_lags["lag_2"]
        row["lag_3"] = current_lags["lag_3"]
        row["lag_7"] = current_lags["lag_7"]
        row["roll_3_mean"] = np.mean(roll_values[-3:]) if len(roll_values) >= 3 else current_lags["lag_1"]
        row["roll_7_mean"] = np.mean(roll_values[-7:]) if len(roll_values) >= 7 else np.mean(roll_values)
        row["roll_3_std"] = np.std(roll_values[-3:]) if len(roll_values) >= 3 else 0

        # Copy weather features from last known row (best available)
        for col in feature_cols:
            if col not in row:
                row[col] = last_row.get(col, 0) if last_row.get(col) is not None else 0

        # Predict
        X = pd.DataFrame([row])[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        pred = float(model.predict(X)[0])
        pred = max(0, pred)  # PM2.5 can't be negative

        # Confidence decay for chained predictions
        if day <= 7:
            confidence = "high"
            confidence_pct = max(70, 95 - (day - 1) * 3)
        elif day <= 15:
            confidence = "medium"
            confidence_pct = max(50, 70 - (day - 7) * 2.5)
        else:
            confidence = "low"
            confidence_pct = max(30, 50 - (day - 15) * 1.5)

        predictions.append({
            "target_date": str(target_date.date()) if hasattr(target_date, 'date') else str(target_date),
            "predicted_pm25": round(pred, 2),
            "horizon_days": day,
            "confidence": confidence,
            "confidence_pct": round(confidence_pct),
        })

        # Update lags for chaining
        current_lags["lag_7"] = current_lags["lag_3"]  # approximate
        current_lags["lag_3"] = current_lags["lag_2"]
        current_lags["lag_2"] = current_lags["lag_1"]
        current_lags["lag_1"] = pred
        roll_values.append(pred)

    return predictions


def run_predictions(conn, run_id):
    """Generate predictions for all countries."""
    all_predictions = {}
    total_predictions = 0

    for cc in COUNTRIES:
        model_path = os.path.join(MODEL_DIR, f"{cc}_pm25_gbr.pkl")
        meta_path = os.path.join(MODEL_DIR, f"{cc}_pm25_meta.json")

        if not os.path.exists(model_path):
            print(f"  ⚠️ No model for {cc}, skipping")
            continue

        model = joblib.load(model_path)
        print(f"\n  {COUNTRY_META[cc]['flag']} {cc}: Generating forecasts...")

        # Get recent features only from active stations.
        df = get_recent_features(conn, cc)
        if df.empty:
            print(f"    No active stations for {cc} in the last {ACTIVE_STATION_MAX_AGE_DAYS} days")
            continue

        # Prefer fresh stations first, then stations with enough context rows.
        station_stats = (
            df.groupby("station_id")
            .agg(rows=("date", "size"), last_date=("date", "max"))
            .sort_values(["last_date", "rows"], ascending=[False, False])
        )
        top_stations = station_stats.head(min(50, len(station_stats))).index.tolist()

        country_preds = []
        for sid in top_stations:
            station_df = df[df["station_id"] == sid].sort_values("date")
            last_row = station_df.iloc[-1].to_dict()
            last_data_date = pd.to_datetime(last_row["date"]).date()
            if (date.today() - last_data_date).days > ACTIVE_STATION_MAX_AGE_DAYS:
                continue

            preds = predict_horizon(model, df, last_row, horizon_days=30, meta_path=meta_path)
            preds = [
                p for p in preds
                if pd.to_datetime(p["target_date"]).date() >= date.today()
            ]
            if not preds:
                continue

            for p in preds:
                p["station_id"] = int(sid)
                p["country"] = cc

            country_preds.extend(preds)

            # Save to prediction_log
            values = [(
                str(run_id), date.today(), cc, int(sid),
                p["target_date"], p["horizon_days"], p["predicted_pm25"]
            ) for p in preds]

            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO prediction_log
                        (run_id, run_date, country_code, station_id,
                         target_date, horizon_days, predicted_value)
                    VALUES %s
                """, values)
            conn.commit()

        if not country_preds:
            print(f"    No valid current/future forecasts for {cc}")
            continue

        # Aggregate country-level forecast (mean of all stations)
        preds_df = pd.DataFrame(country_preds)
        daily_agg = preds_df.groupby(["target_date", "horizon_days", "confidence", "confidence_pct"]).agg(
            mean_pm25=("predicted_pm25", "mean"),
            min_pm25=("predicted_pm25", lambda x: np.percentile(x, 10)),
            max_pm25=("predicted_pm25", lambda x: np.percentile(x, 90)),
            stations=("station_id", "nunique"),
        ).reset_index()

        country_forecast = {
            "country": cc,
            "meta": COUNTRY_META[cc],
            "generated_at": datetime.now().isoformat(),
            "last_data_date": str(df["date"].max()),
            "forecast": daily_agg.to_dict(orient="records"),
            "station_count": len(top_stations),
        }

        all_predictions[cc] = country_forecast
        total_predictions += len(country_preds)
        print(f"    {len(top_stations)} stations × 30 days = {len(country_preds)} predictions")

    return all_predictions, total_predictions


def export_site_data(predictions, metric_mae, metric_r2, metric_source,
                     metric_sample_count, live_validation_count):
    """Export JSON files for the Vercel static site."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Per-country prediction files
    for cc, data in predictions.items():
        path = os.path.join(OUTPUT_DIR, f"predictions_{cc}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Combined metadata
    model_meta = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "v5",
        "countries": {},
        "accuracy": {
            "mae": round(metric_mae, 2) if metric_mae is not None else None,
            "r2": round(metric_r2, 4) if metric_r2 is not None else None,
            "source": metric_source,
            "sample_count": metric_sample_count,
            "live_validation_count": live_validation_count,
            "note": (
                "Live metrics require 100+ validated completed forecasts. "
                "Backtest metrics are recent one-step checks against known actuals."
            ),
        }
    }
    for cc in COUNTRIES:
        meta = COUNTRY_META[cc].copy()
        if cc in predictions:
            meta["station_count"] = predictions[cc]["station_count"]
            meta["last_data_date"] = predictions[cc]["last_data_date"]
            meta["forecast_days"] = 30
        model_meta["countries"][cc] = meta

    with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)

    # Accuracy data for the proof section
    accuracy = {
        "generated_at": datetime.now().isoformat(),
        "mae": round(metric_mae, 2) if metric_mae is not None else None,
        "r2": round(metric_r2, 4) if metric_r2 is not None else None,
        "source": metric_source,
        "sample_count": metric_sample_count,
        "live_validation_count": live_validation_count,
        "training_metrics": {
            cc: {"r2": m["test_r2"], "mae": m["test_mae"]}
            for cc, m in COUNTRY_META.items()
        },
        "confidence_explanation": {
            "7_day": "Direct model prediction using real lag features. Highest accuracy.",
            "15_day": "Chained predictions — each day uses predicted values as inputs. Accuracy decreases.",
            "30_day": "Extended chained forecast. Treat as directional trend only.",
        }
    }
    with open(os.path.join(OUTPUT_DIR, "accuracy.json"), "w") as f:
        json.dump(accuracy, f, indent=2)

    print(f"\n  Exported to {OUTPUT_DIR}/")
    print(f"    predictions_IN.json, predictions_US.json, ...")
    print(f"    model_meta.json, accuracy.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching, just predict")
    args = parser.parse_args()

    start = time.time()
    run_id = uuid.uuid4()
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_tracking_schema(conn)

    print("═" * 60)
    print("  GLOBAL AQ INTELLIGENCE — PREDICTION PIPELINE")
    print(f"  Run ID: {run_id}")
    print(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 60)

    # Record run start
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_runs (run_id, run_date, status)
            VALUES (%s, NOW(), 'running')
        """, (str(run_id),))
    conn.commit()

    # Phase 1: Check last run
    last_run = get_last_run_date(conn)
    gap_days = (date.today() - last_run).days if last_run else None
    print(f"\n  Last run: {last_run or 'never'}")
    if gap_days:
        print(f"  Gap: {gap_days} days")

    # Phase 2: Validate old predictions
    print(f"\n{'─'*60}")
    print("  Phase 1: Validate Old Predictions")
    print(f"{'─'*60}")
    validated, live_mae, live_r2, live_validation_count = validate_old_predictions(conn, run_id)

    # Phase 3: Generate new predictions
    print(f"\n{'─'*60}")
    print("  Phase 2: Generate New Forecasts (7d + 15d + 30d)")
    print(f"{'─'*60}")
    predictions, total_preds = run_predictions(conn, run_id)

    # Phase 3.5: Backtest — predict last 7 days vs actuals for fresh metrics
    print(f"\n{'─'*60}")
    print("  Phase 2.5: Backtest (Last 7 Days vs Actuals)")
    print(f"{'─'*60}")
    bt_mae, bt_r2, bt_sample_count, bt_country = backtest_recent(conn, n_days=7)

    # Phase 4: Export JSON for site
    print(f"\n{'─'*60}")
    print("  Phase 3: Export Site Data")
    print(f"{'─'*60}")

    # Priority: live validated > backtest > none, with source shown honestly.
    if live_mae is not None:
        display_mae = live_mae
        display_r2 = live_r2
        metrics_source = "live"
        metric_sample_count = live_validation_count
    elif bt_mae is not None:
        display_mae = bt_mae
        display_r2 = bt_r2
        metrics_source = "backtest"
        metric_sample_count = bt_sample_count
    else:
        display_mae = None
        display_r2 = None
        metrics_source = "none"
        metric_sample_count = 0

    export_site_data(
        predictions,
        display_mae,
        display_r2,
        metrics_source,
        metric_sample_count,
        live_validation_count,
    )

    # Update run record
    elapsed = time.time() - start

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE pipeline_runs SET
                status = 'completed',
                predictions_made = %s,
                validations_done = %s,
                live_mae = %s,
                live_r2 = %s,
                backtest_mae = %s,
                backtest_r2 = %s,
                metric_source = %s,
                metric_sample_count = %s
            WHERE run_id = %s
        """, (
            total_preds,
            validated,
            live_mae,
            live_r2,
            bt_mae,
            bt_r2,
            metrics_source,
            metric_sample_count,
            str(run_id),
        ))
    conn.commit()

    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE ({int(elapsed)}s)")
    print(f"{'═'*60}")
    print(f"  Predictions: {total_preds:,}")
    print(f"  Validated:   {validated}")
    if display_mae is not None:
        label = {"live": "Live", "backtest": "Backtest", "none": ""}[metrics_source]
        print(f"  {label} MAE:  {display_mae:.2f} µg/m³")
        print(f"  {label} R²:   {display_r2:.4f}")
        print(f"  Samples:     {metric_sample_count:,}")

    conn.close()


if __name__ == "__main__":
    main()
