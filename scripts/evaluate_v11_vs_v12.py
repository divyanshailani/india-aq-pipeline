import os
import sys
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from src.evaluation import calculate_mae, calculate_nmae, calculate_mase, calculate_accuracy
from scripts.train_v11_aod_global import build_v11_features

import glob

def load_recent_v11_data(conn, country_code):
    sql = """
        SELECT df.station_id, df.date, df.value,
               df.om_temperature as future_temp,
               df.om_wind_speed as future_wind,
               df.om_precipitation as future_precip,
               0.0 as wind_direction,
               df.rolling_3day_precip,
               df.aod_volatility_index,
               s.latitude as station_lat,
               s.longitude as station_lon,
               aod.aod_mean,
               aod.aod_max
        FROM daily_features df
        JOIN stations s ON df.station_id = s.id
        LEFT JOIN satellite_aod_features aod 
            ON df.station_id = aod.station_id AND df.date = aod.date
        WHERE df.country_code = %s
          AND df.parameter = 'pm25'
          AND df.value IS NOT NULL
          AND df.date >= '2025-01-01'
        ORDER BY df.station_id, df.date
    """
    df = pd.read_sql(sql, conn, params=(country_code,))
    df["date"] = pd.to_datetime(df["date"])
    
    # Reconstruct wind_u and wind_v since wind_direction is no longer available in DB
    df["wind_u"] = -df["future_wind"] * np.sin(np.radians(df["wind_direction"]))
    df["wind_v"] = -df["future_wind"] * np.cos(np.radians(df["wind_direction"]))
    
    try:
        csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "DL_FIRE_SV-C2_*", "fire_*.csv"))
        viirs_list = []
        bounds = {
            "IN": (6, 38, 68, 98),
            "US": (24, 50, -125, -66),
            "GB": (49, 61, -9, 2),
            "AU": (-44, -10, 112, 154)
        }
        min_lat, max_lat, min_lon, max_lon = bounds.get(country_code, (-90, 90, -180, 180))
        for f in csv_files:
            v = pd.read_csv(f, usecols=["latitude", "longitude", "brightness", "acq_date"])
            v = v[(v["latitude"] >= min_lat) & (v["latitude"] <= max_lat) & 
                  (v["longitude"] >= min_lon) & (v["longitude"] <= max_lon)]
            v = v.rename(columns={"latitude": "fire_lat", "longitude": "fire_lon"})
            viirs_list.append(v)
            
        if viirs_list:
            viirs = pd.concat(viirs_list, ignore_index=True)
            viirs["acq_date"] = pd.to_datetime(viirs["acq_date"])
        else:
            viirs = pd.DataFrame(columns=["fire_lat", "fire_lon", "brightness", "acq_date"])
    except Exception as e:
        viirs = pd.DataFrame(columns=["fire_lat", "fire_lon", "brightness", "acq_date"])
        
    return df, viirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

V11_MODEL_DIR = os.path.join("models", "v11")
V12_MODEL_DIR = os.path.join("models", "v12")
PREDICTIONS_DIR = os.path.join("data", "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

HOLDOUT_START = "2026-01-01"

def main():
    logging.info("Starting Champion (V11.1) vs Challenger (V12) Evaluation...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    logging.info("Loading V12 Parquet data...")
    df_v12_base = pd.read_parquet("data/daily_features_full.parquet")
    
    countries = ["US", "GB", "IN", "AU"]
    horizons = [1, 7, 14, 30]
    
    results = []
    
    for country in countries:
        logging.info(f"========== {country} ==========")
        df_v11_raw, viirs = load_recent_v11_data(conn, country)
        df_v12_country = df_v12_base[df_v12_base['country_code'] == country].copy()
        df_v12_country['date'] = pd.to_datetime(df_v12_country['date'])
        
        for horizon in horizons:
            logging.info(f"--- Horizon {horizon}d ---")
            
            # --- V11 Prep ---
            df_v11_h, _ = build_v11_features(df_v11_raw, viirs, horizon)
            df_v11_h['date'] = pd.to_datetime(df_v11_h['date'])
            df_v11_h = df_v11_h.dropna(subset=['value'])
            
            # V11 Model
            v11_meta_path = os.path.join(V11_MODEL_DIR, f"{country}_pm25_h{horizon}_meta.json")
            v11_xgb_path = os.path.join(V11_MODEL_DIR, f"{country}_pm25_h{horizon}_xgb.json")
            if not os.path.exists(v11_xgb_path):
                logging.warning(f"V11 model missing for {country} h={horizon}. Skipping.")
                continue
                
            with open(v11_meta_path) as f:
                v11_meta = json.load(f)
            v11_features = v11_meta["features"]
            
            # Filter V11 for holdout and drop NAs in features
            v11_test = df_v11_h[df_v11_h['date'] >= HOLDOUT_START].copy()
            v11_test = v11_test.dropna(subset=v11_features)
            
            if len(v11_test) == 0:
                logging.warning(f"No V11 holdout data for {country} h={horizon}")
                continue
                
            v11_model = xgb.XGBRegressor()
            v11_model.load_model(v11_xgb_path)
            v11_test['v11_pred'] = v11_model.predict(v11_test[v11_features])
            
            # For naive baseline (MASE), V11 uses pm25_lag_{horizon} as the naive prediction
            v11_test['naive_pred'] = v11_test[f"pm25_lag_{horizon}"]
            
            # Trim V11 columns
            res_v11 = v11_test[['date', 'station_id', 'value', 'v11_pred', 'naive_pred']].copy()
            res_v11.rename(columns={'value': 'actual'}, inplace=True)
            
            # --- V12 Prep ---
            df_v12_h = df_v12_country.copy()
            for h_temp in [1, 7, 14, 30]:
                df_v12_h[f'target_{h_temp}d'] = df_v12_h.groupby('station_id')['value'].shift(-h_temp)
            
            df_v12_h['target_date'] = df_v12_h['date'] + pd.Timedelta(days=horizon)
            
            # Drop NAs for the specific horizon target
            df_v12_h = df_v12_h.dropna(subset=[f'target_{horizon}d'])
            
            v12_model_dir = os.path.join(V12_MODEL_DIR, country, f"horizon_{horizon}")
            v12_xgb_path = os.path.join(v12_model_dir, "model.json")
            if not os.path.exists(v12_xgb_path):
                logging.warning(f"V12 model missing for {country} h={horizon}. Skipping.")
                continue
                
            v12_model = xgb.XGBRegressor()
            v12_model.load_model(v12_xgb_path)
            v12_features = v12_model.get_booster().feature_names
            
            # Filter for holdout and drop NAs in features
            v12_test = df_v12_h[df_v12_h['target_date'] >= HOLDOUT_START].copy()
            v12_test = v12_test.dropna(subset=v12_features)
            
            if len(v12_test) == 0:
                logging.warning(f"No V12 holdout data for {country} h={horizon}")
                continue
                
            v12_test['v12_pred'] = v12_model.predict(v12_test[v12_features])
            
            # Trim V12 columns (map target_date to date for merging with V11)
            res_v12 = v12_test[['target_date', 'station_id', 'v12_pred']].copy()
            res_v12.rename(columns={'target_date': 'date'}, inplace=True)
            
            # --- INNER MERGE FOR FLAWLESS APPLES-TO-APPLES COMPARISON ---
            merged = pd.merge(res_v11, res_v12, on=['date', 'station_id'], how='inner')
            
            if len(merged) == 0:
                logging.warning(f"No overlapping test data for {country} h={horizon}. V11: {len(res_v11)}, V12: {len(res_v12)}")
                continue
                
            y_actual = merged['actual'].values
            y_v11 = merged['v11_pred'].values
            y_v12 = merged['v12_pred'].values
            y_naive = merged['naive_pred'].values
            
            # Compute Metrics for V11
            v11_mae = calculate_mae(y_actual, y_v11)
            v11_nmae = calculate_nmae(y_actual, y_v11)
            v11_mase = calculate_mase(y_actual, y_v11, y_naive)
            v11_acc = calculate_accuracy(v11_nmae)
            
            # Compute Metrics for V12
            v12_mae = calculate_mae(y_actual, y_v12)
            v12_nmae = calculate_nmae(y_actual, y_v12)
            v12_mase = calculate_mase(y_actual, y_v12, y_naive)
            v12_acc = calculate_accuracy(v12_nmae)
            
            # Determine Winner
            winner = "V12" if v12_mae < v11_mae else "V11.1"
            
            results.append({
                "Country": country,
                "Horizon": f"{horizon}d",
                "Samples": len(merged),
                "V11_MAE": v11_mae,
                "V12_MAE": v12_mae,
                "V11_NMAE": v11_nmae,
                "V12_NMAE": v12_nmae,
                "V11_MASE": v11_mase,
                "V12_MASE": v12_mase,
                "V11_Acc": v11_acc,
                "V12_Acc": v12_acc,
                "Winner": winner
            })
            
            # Save predictions for plotting
            pred_file = os.path.join(PREDICTIONS_DIR, f"holdout_eval_{country}_{horizon}.csv")
            merged.to_csv(pred_file, index=False)
            logging.info(f"Saved {len(merged)} overlapping predictions to {pred_file}")

    # Generate Markdown Table
    df_res = pd.DataFrame(results)
    if df_res.empty:
        logging.error("No results generated!")
        return
        
    print("\n# Champion vs Challenger: First-Principles Evaluation")
    print(f"**Holdout Period:** {HOLDOUT_START} to Latest")
    print("| Country | Horizon | Samples | V11 MAE | V12 MAE | V11 NMAE | V12 NMAE | V11 MASE | V12 MASE | V11 Acc | V12 Acc | WINNER |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    for _, row in df_res.iterrows():
        print(f"| {row['Country']} | {row['Horizon']} | {row['Samples']:,} "
              f"| {row['V11_MAE']:.2f} | **{row['V12_MAE']:.2f}** "
              f"| {row['V11_NMAE']:.3f} | **{row['V12_NMAE']:.3f}** "
              f"| {row['V11_MASE']:.3f} | **{row['V12_MASE']:.3f}** "
              f"| {row['V11_Acc']:.1f}% | **{row['V12_Acc']:.1f}%** "
              f"| **{row['Winner']}** |")
              
    # Summary of Winner
    v12_wins = sum(df_res['Winner'] == 'V12')
    total = len(df_res)
    print(f"\n**Final Verdict:** V12 wins {v12_wins} out of {total} matchups.")

if __name__ == "__main__":
    main()
