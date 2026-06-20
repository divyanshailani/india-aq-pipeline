import os
import sys
import json
import psycopg2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from train_v9_4_xgboost import load_base_data, build_v9_4_features, temporal_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v9_4")
HORIZONS = [1, 7, 14, 30]

def plot_best_station():
    print("Connecting to DB...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Plot for India (IN)
    df_in, viirs_data = load_base_data(conn, "IN")
    
    # Get station names
    sql = "SELECT id as station_id, name as station_name FROM stations WHERE country_code = 'IN'"
    stations_df = pd.read_sql(sql, conn)
    conn.close()
    
    df_in = df_in.merge(stations_df, on="station_id", how="left")
    
    counts = df_in.groupby(["station_id", "station_name"]).size().reset_index(name="days")
    best_station = counts.sort_values("days", ascending=False).iloc[0]
    best_sid = best_station["station_id"]
    best_name = best_station["station_name"]
    print(f"Selected station: {best_name} ({best_sid}) with {best_station['days']} days")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, h in enumerate(HORIZONS):
        print(f"Processing horizon {h}...")
        df_h, features = build_v9_4_features(df_in, viirs_data, h)
        
        df_station = df_h[df_h["station_id"] == best_sid].copy()
        
        if len(df_station) < 50:
            print(f"Not enough data for horizon {h}")
            continue
            
        train, test = temporal_split(df_station, "value", test_ratio=0.20)
        if test.empty:
            continue
            
        model_path = os.path.join(MODEL_DIR, f"IN_pm25_h{h}_xgb.json")
        meta_path = os.path.join(MODEL_DIR, f"IN_pm25_h{h}_meta.json")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
        medians = meta["feature_medians"]
        
        X_test = test[features].copy()
        y_test = test["value"].values
        y_test_delta = test["value"].values - test[f"pm25_lag_{h}"].values
        
        for col in features:
            X_test[col] = X_test[col].fillna(medians.get(col, 0.0))
            
        X_test = X_test.replace([float("inf"), float("-inf")], 0)
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        y_pred_delta = model.predict(X_test)
        y_pred = y_pred_delta + test[f"pm25_lag_{h}"].values
        y_pred = [max(0, p) for p in y_pred] 
        
        mae = mean_absolute_error(y_test, y_pred)
        
        ax = axes[i // 2, i % 2]
        
        plot_len = min(60, len(y_test))
        y_test_plot = y_test[-plot_len:]
        y_pred_plot = y_pred[-plot_len:]
        
        ax.plot(range(len(y_test_plot)), y_test_plot, 'b-', label='Actual', linewidth=1.5)
        ax.plot(range(len(y_pred_plot)), y_pred_plot, 'r--', label='Predicted (V9.4 Delta)', linewidth=1.5)
        
        ax.set_title(f'{h}-Day Forecast (MAE={mae:.1f})')
        ax.set_xlabel('Days (Test Split)')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.legend()

    plt.suptitle(f'V9.4 Geospatial Ensemble Forecast: {best_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'forecast_horizons_v9_4.png')
    plt.savefig(out_path, dpi=150)
    print(f"\\nPlot saved: {out_path}")

if __name__ == "__main__":
    plot_best_station()
