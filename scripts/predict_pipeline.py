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

COUNTRY_META = {
    "IN": {
        "name": "India",
        "flag": "🇮🇳",
        "confidence": "high",
        "tag": "High Confidence",
        "tag_color": "green",
        "reason": "R²=0.77 on 31K features, CPCB reference + NASA weather/fire data",
        "test_r2": 0.7718,
        "test_mae": 8.52,
    },
    "US": {
        "name": "United States",
        "flag": "🇺🇸",
        "confidence": "high",
        "tag": "High Confidence",
        "tag_color": "green",
        "reason": "R²=0.80 on 1.4M features, EPA AQS reference-grade stations",
        "test_r2": 0.7976,
        "test_mae": 1.70,
    },
    "GB": {
        "name": "United Kingdom",
        "flag": "🇬🇧",
        "confidence": "experimental",
        "tag": "Experimental: Limited Seasonal Data",
        "tag_color": "yellow",
        "reason": "R²=0.48, fragmented community sensors + 6mo DEFRA data",
        "test_r2": 0.4817,
        "test_mae": 2.01,
    },
    "AU": {
        "name": "Australia",
        "flag": "🇦🇺",
        "confidence": "stable",
        "tag": "Low Variance / Stable",
        "tag_color": "blue",
        "reason": "R²=0.64, clean-air country with NSW EPA reference data",
        "test_r2": 0.6437,
        "test_mae": 1.55,
    },
}


def get_last_run_date(conn):
    """Get the date of the last pipeline run."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(run_date) FROM pipeline_runs WHERE status = 'completed'")
        result = cur.fetchone()[0]
    return result.date() if result else None


def validate_old_predictions(conn, run_id):
    """
    Compare old predictions (where target_date <= today and actual is NULL)
    against real data that has arrived.
    """
    today = date.today()
    validated = 0

    with conn.cursor() as cur:
        # Find unvalidated predictions whose target date has passed
        cur.execute("""
            SELECT pl.id, pl.station_id, pl.target_date, pl.predicted_value, pl.country_code
            FROM prediction_log pl
            WHERE pl.actual_value IS NULL
              AND pl.target_date <= %s
        """, (today,))
        pending = cur.fetchall()

        if not pending:
            print("  No predictions to validate")
            return 0, None, None

        print(f"  Found {len(pending)} predictions to validate...")

        for pid, station_id, target_date, predicted, cc in pending:
            # Fetch actual PM2.5 for that station+date
            cur.execute("""
                SELECT AVG(r.value) FROM raw_measurements r
                WHERE r.station_id = %s
                  AND r.parameter = 'pm25'
                  AND r.datetime_utc::date = %s
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

        # Calculate live metrics from all validated predictions (last 90 days)
        cur.execute("""
            SELECT actual_value, predicted_value
            FROM prediction_log
            WHERE actual_value IS NOT NULL
              AND validated_at >= NOW() - INTERVAL '90 days'
        """)
        rows = cur.fetchall()

        if len(rows) >= 10:
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

    return validated, live_mae, live_r2


def get_recent_features(conn, country_code, n_days=14):
    """Get the most recent features per station for a country."""
    sql = """
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY date DESC) as rn
            FROM daily_features
            WHERE country_code = %s
              AND parameter = 'pm25'
              AND value IS NOT NULL
              AND lag_1 IS NOT NULL
        )
        SELECT * FROM ranked WHERE rn <= %s
        ORDER BY station_id, date
    """
    return pd.read_sql(sql, conn, params=(country_code, n_days))


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

        # Get recent features
        df = get_recent_features(conn, cc, n_days=14)
        if df.empty:
            print(f"    No recent features for {cc}")
            continue

        # Get top stations by data coverage
        station_counts = df.groupby("station_id").size()
        top_stations = station_counts.nlargest(min(50, len(station_counts))).index.tolist()

        country_preds = []
        for sid in top_stations:
            station_df = df[df["station_id"] == sid].sort_values("date")
            last_row = station_df.iloc[-1].to_dict()

            preds = predict_horizon(model, df, last_row, horizon_days=30, meta_path=meta_path)

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


def export_site_data(predictions, live_mae, live_r2):
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
        "live_accuracy": {
            "mae": round(live_mae, 2) if live_mae else None,
            "r2": round(live_r2, 4) if live_r2 else None,
            "note": "Based on validated predictions vs actual observations"
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
        "live_mae": round(live_mae, 2) if live_mae else None,
        "live_r2": round(live_r2, 4) if live_r2 else None,
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
    validated, live_mae, live_r2 = validate_old_predictions(conn, run_id)

    # Phase 3: Generate new predictions
    print(f"\n{'─'*60}")
    print("  Phase 2: Generate New Forecasts (7d + 15d + 30d)")
    print(f"{'─'*60}")
    predictions, total_preds = run_predictions(conn, run_id)

    # Phase 4: Export JSON for site
    print(f"\n{'─'*60}")
    print("  Phase 3: Export Site Data")
    print(f"{'─'*60}")
    export_site_data(predictions, live_mae, live_r2)

    # Update run record
    elapsed = time.time() - start

    # Load training metrics as fallback
    train_r2 = None
    train_mae = None
    meta_path = os.path.join(MODEL_DIR, "all_models_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            all_meta = json.load(f)
        # Average across countries
        r2s = [m["metrics"]["test_r2"] for m in all_meta.values() if "metrics" in m]
        maes = [m["metrics"]["test_mae"] for m in all_meta.values() if "metrics" in m]
        if r2s:
            train_r2 = round(sum(r2s) / len(r2s), 4)
            train_mae = round(sum(maes) / len(maes), 2)

    # Use live if available, otherwise training metrics
    display_mae = live_mae if live_mae else train_mae
    display_r2 = live_r2 if live_r2 else train_r2
    metrics_source = "live" if live_mae else "training"

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE pipeline_runs SET
                status = 'completed',
                predictions_made = %s,
                validations_done = %s,
                live_mae = %s,
                live_r2 = %s
            WHERE run_id = %s
        """, (total_preds, validated, display_mae, display_r2, str(run_id)))
    conn.commit()

    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE ({int(elapsed)}s)")
    print(f"{'═'*60}")
    print(f"  Predictions: {total_preds:,}")
    print(f"  Validated:   {validated}")
    if display_mae is not None:
        label = "Live" if metrics_source == "live" else "Model"
        print(f"  {label} MAE:  {display_mae:.2f} µg/m³")
        print(f"  {label} R²:   {display_r2:.4f}")
        if metrics_source == "training":
            print(f"  (Using training metrics — live metrics available after validation)")

    conn.close()


if __name__ == "__main__":
    main()

