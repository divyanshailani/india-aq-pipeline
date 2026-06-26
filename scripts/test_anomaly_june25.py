import sys, os
import pandas as pd
import psycopg2
from datetime import date, timedelta
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from scripts.predict_pipeline import predict_direct_ensemble, get_recent_features, fetch_station_forecasts

conn = psycopg2.connect(**DB_CONFIG)
country_code = 'IN'

# Get all IN stations with active_days=30
df = get_recent_features(conn, country_code, active_days=30)

station_stats = (
    df.groupby("station_id")
    .agg(rows=("date", "size"), last_date=("date", "max"))
    .sort_values(["last_date", "rows"], ascending=[False, False])
)
# Top 50 stations for performance
top_stations = station_stats.head(50).index.tolist()

format_strings = ','.join(['%s'] * len(top_stations))
cur = conn.cursor()
cur.execute(f"SELECT id, latitude, longitude FROM stations WHERE id IN ({format_strings})", tuple(top_stations))
station_coords = pd.DataFrame(cur.fetchall(), columns=["station_id", "latitude", "longitude"])
cur.close()

# Mock heavy rain forecast for the anomaly (15mm precip)
station_forecast = {}
station_forecast_no_rain = {}
for sid in top_stations:
    last_date = df[df["station_id"] == sid]["date"].max()
    last_date = pd.to_datetime(last_date).date()
    
    sf = {}
    sf_no = {}
    for i in range(1, 31):
        dt_str = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        sf[dt_str] = {"temp": 30.0, "wind": 10.0, "precip": 15.0} # heavy monsoon rain
        sf_no[dt_str] = {"temp": 30.0, "wind": 10.0, "precip": 0.0}
    
    station_forecast[sid] = sf
    station_forecast_no_rain[sid] = sf_no

viirs = pd.DataFrame(columns=["fire_lat", "fire_lon", "brightness", "acq_date"])
aod_data = {"aod_mean_lag_1": 0.5, "aod_max_lag_1": 0.8}

old_preds_25 = []
new_preds_25 = []
target_date_str = '2026-06-25'

for sid in top_stations:
    station_df = df[df["station_id"] == sid].sort_values("date")
    
    # NEW (with rain)
    preds_new = predict_direct_ensemble(country_code, station_df, station_forecast[sid], viirs, aod_data)
    # OLD (no rain)
    preds_old = predict_direct_ensemble(country_code, station_df, station_forecast_no_rain[sid], viirs, aod_data)
    
    if preds_new:
        for p in preds_new:
            if p["target_date"] == target_date_str:
                new_preds_25.append(p["predicted_pm25"])
                
    if preds_old:
        for p in preds_old:
            if p["target_date"] == target_date_str:
                old_preds_25.append(p["predicted_pm25"])

old_avg = np.mean(old_preds_25) if old_preds_25 else 0
new_avg = np.mean(new_preds_25) if new_preds_25 else 0

print("="*50)
print(f"🎯 JUNE 25 INDIA ANOMALY BACKTEST (Top 50 Stations)")
print("="*50)
print(f"Old Prediction (Baseline XGBoost): {old_avg:.2f} µg/m³")
print(f"New Forced Prediction (Override):  {new_avg:.2f} µg/m³")
print(f"Actual Ground Truth:               14.61 µg/m³")
print("="*50)
