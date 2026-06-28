"""
V12 ONNX Inference Engine for Azure VM
======================================
Reads `data/daily_features_full.parquet` directly.
Runs ONNX inference for 16 models (4 countries × 4 horizons).
Interpolates intermediate days (1-30).
Outputs Next.js compatible static JSON directly.
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import onnxruntime as ort

# V12 Constants
V12_FEATURES = [
    'month', 'day_of_week', 'is_weekend', 'day_of_year', 
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30', 
    'roll_3_mean', 'roll_7_mean', 'roll_3_std', 'roll_14_mean', 'roll_30_mean', 'roll_14_std', 
    'om_temperature', 'om_wind_speed', 'om_precipitation', 'om_aerosol_optical_depth', 
    'rolling_3day_precip', 'aod_volatility_index', 'latitude', 'longitude'
]

COUNTRIES = ['AU', 'GB', 'IN', 'US']
HORIZONS = [1, 7, 14, 30]

COUNTRY_META = {
    "IN": {"name": "India", "flag": "🇮🇳", "anchor": "Delhi", "confidence": "high", "tag": "V12 Engine", "tag_color": "green", "reason": "V12 ONNX Inference", "accuracy_percentage": 50.0, "test_mae": 27.1},
    "US": {"name": "United States", "flag": "🇺🇸", "anchor": "Washington D.C.", "confidence": "high", "tag": "V12 Engine", "tag_color": "green", "reason": "V12 ONNX Inference", "accuracy_percentage": 85.0, "test_mae": 2.5},
    "GB": {"name": "United Kingdom", "flag": "🇬🇧", "anchor": "London", "confidence": "high", "tag": "V12 Engine", "tag_color": "green", "reason": "V12 ONNX Inference", "accuracy_percentage": 88.0, "test_mae": 1.5},
    "AU": {"name": "Australia", "flag": "🇦🇺", "anchor": "Canberra", "confidence": "high", "tag": "V12 Engine", "tag_color": "green", "reason": "V12 ONNX Inference", "accuracy_percentage": 80.0, "test_mae": 3.0},
}

def load_models(model_dir):
    """Loads all 16 ONNX models into a dictionary."""
    sessions = {}
    for cc in COUNTRIES:
        sessions[cc] = {}
        for h in HORIZONS:
            model_path = os.path.join(model_dir, cc, f"horizon_{h}", "model.onnx")
            if os.path.exists(model_path):
                sessions[cc][h] = ort.InferenceSession(model_path)
            else:
                print(f"Warning: Model not found at {model_path}")
    return sessions

def get_latest_station_features(df, country):
    """Gets the most recent row for each station in a country."""
    df_c = df[df['country_code'] == country]
    if df_c.empty:
        return pd.DataFrame()
    
    # Sort by date descending and drop duplicates by station_id
    latest_df = df_c.sort_values('date', ascending=False).drop_duplicates(subset=['station_id'])
    return latest_df

def run_inference():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'daily_features_full.parquet')
    model_dir = os.path.join(base_dir, 'models', 'v12')
    output_dir = os.path.join(base_dir, 'site_data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    if 'parameter' in df.columns:
        df = df[df['parameter'] == 'pm25']
        
    print(f"Loading 16 ONNX models...")
    sessions = load_models(model_dir)
    
    predictions = {}
    
    for cc in COUNTRIES:
        print(f"Processing {cc}...")
        latest_df = get_latest_station_features(df, cc)
        if latest_df.empty:
            print(f"No data for {cc}")
            continue
            
        last_data_date_str = str(latest_df['date'].max())
        last_data_date = pd.to_datetime(last_data_date_str).date()
        
        # Ensure all V12 features exist and fill NaNs with 0 (XGBoost hist handles NaNs natively, 
        # but for ONNX we need to feed numbers. Wait, ONNX handles NaNs as float('nan'))
        X = latest_df[V12_FEATURES].astype(np.float32).copy()
        
        station_ids = latest_df['station_id'].tolist()
        num_stations = len(station_ids)
        
        # Predict 4 anchors
        anchors = {}
        for h in HORIZONS:
            sess = sessions[cc].get(h)
            if not sess:
                anchors[h] = np.zeros(num_stations)
                continue
                
            input_name = sess.get_inputs()[0].name
            preds = sess.run(None, {input_name: X.values})[0].flatten()
            anchors[h] = preds
            
        # Interpolate for 1..30 days
        forecast_records = []
        for day in range(1, 31):
            if day in HORIZONS:
                day_preds = anchors[day]
            elif day < HORIZONS[0]:
                day_preds = anchors[HORIZONS[0]]
            elif day > HORIZONS[-1]:
                day_preds = anchors[HORIZONS[-1]]
            else:
                left = max(a for a in HORIZONS if a < day)
                right = min(a for a in HORIZONS if a > day)
                weight = (day - left) / (right - left)
                day_preds = anchors[left] + weight * (anchors[right] - anchors[left])
                
            # Prevent negative predictions
            day_preds = np.maximum(day_preds, 0.0)
            
            mean_pm25 = float(np.mean(day_preds))
            min_pm25 = float(np.percentile(day_preds, 10))
            max_pm25 = float(np.percentile(day_preds, 90))
            
            target_date = last_data_date + timedelta(days=day)
            
            if day <= 7:
                conf, conf_pct = "high", max(70, 95 - (day - 1) * 3)
            elif day <= 15:
                conf, conf_pct = "medium", max(50, 70 - (day - 7) * 2.5)
            else:
                conf, conf_pct = "low", max(30, 50 - (day - 15) * 1.5)
                
            forecast_records.append({
                "target_date": str(target_date),
                "horizon_days": day,
                "confidence": conf,
                "confidence_pct": round(conf_pct),
                "mean_pm25": round(mean_pm25, 2),
                "min_pm25": round(min_pm25, 2),
                "max_pm25": round(max_pm25, 2),
                "stations": num_stations
            })
            
        predictions[cc] = {
            "country": cc,
            "meta": {
                **COUNTRY_META[cc],
                "fire_count": 0
            },
            "generated_at": datetime.now().isoformat(),
            "last_data_date": last_data_date_str,
            "forecast": forecast_records,
            "station_count": num_stations
        }
        
    # Export to site_data
    for cc, data in predictions.items():
        path = os.path.join(output_dir, f"predictions_{cc}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
    # Metadata
    model_meta = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "v12_onnx_global",
        "countries": {cc: COUNTRY_META[cc] for cc in COUNTRIES}
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)
        
    print(f"Exported JSON outputs to {output_dir}/")

if __name__ == "__main__":
    run_inference()
