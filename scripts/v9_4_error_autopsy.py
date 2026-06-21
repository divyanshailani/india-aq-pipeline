import os
import sys
import json
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from scripts.train_v9_4_xgboost import load_base_data, build_v9_4_features, temporal_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v9_4")

def main():
    print("Initializing V9.4 Autopsy for IN (h=1)...")
    
    # 1. Load Data
    conn = psycopg2.connect(**DB_CONFIG)
    df_in, viirs_data = load_base_data(conn, "IN")
    conn.close()
    
    # 2. Build Features
    df_h, features = build_v9_4_features(df_in, viirs_data, 1)
    
    # 3. Split
    train, test = temporal_split(df_h, "value", test_ratio=0.20)
    
    # 4. Load Model
    model_path = os.path.join(MODEL_DIR, "IN_pm25_h1_xgb.json")
    meta_path = os.path.join(MODEL_DIR, "IN_pm25_h1_meta.json")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    medians = meta["feature_medians"]
    
    X_test = test[feature_cols].copy()
    y_test_true = test["value"].values
    y_test_lag = test["pm25_lag_1"].values
    
    # Impute
    for col in feature_cols:
        X_test[col] = X_test[col].fillna(medians.get(col, 0.0))
    X_test = X_test.replace([float("inf"), float("-inf")], 0)
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # 5. Predict & Reconstruct
    delta_pred = model.predict(X_test)
    y_pred = delta_pred + y_test_lag
    y_pred = np.maximum(0, y_pred)
    
    # 6. Calculate Absolute Error
    test["predicted"] = y_pred
    test["abs_error"] = np.abs(test["value"] - test["predicted"])
    
    print("\n" + "="*50)
    print(f"Overall Mean Absolute Error: {test['abs_error'].mean():.2f} µg/m³")
    print("="*50)
    
    # 7. Diagnostic Slices
    
    # Temporal Slice: Day of Week
    print("\n--- 1. Temporal Slice (Day of Week) ---")
    test["date"] = pd.to_datetime(test["date"])
    test["day_of_week"] = test["date"].dt.day_name()
    dow_error = test.groupby("day_of_week")["abs_error"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    for dow, err in dow_error.items():
        print(f"{dow.ljust(10)}: {err:.2f} µg/m³")
        
    # Magnitude Slice
    print("\n--- 2. Magnitude Slice (True PM2.5 Buckets) ---")
    bins = [0, 50, 150, float('inf')]
    labels = ["Low (< 50)", "Medium (50-150)", "High (> 150)"]
    test["magnitude_bucket"] = pd.cut(test["value"], bins=bins, labels=labels)
    mag_error = test.groupby("magnitude_bucket")["abs_error"].agg(["mean", "count"])
    for bucket, row in mag_error.iterrows():
        print(f"{str(bucket).ljust(15)}: Error = {row['mean']:.2f} µg/m³ (N={int(row['count'])})")
        
    # Spatial/Fire Slice
    print("\n--- 3. Spatial/Fire Slice (VIIRS Impact) ---")
    if "fire_density_100km" in test.columns:
        test["fire_bucket"] = pd.cut(test["fire_density_100km"], bins=[-1, 0, 5, float('inf')], labels=["No Fires (0)", "Low Fires (1-5)", "High Fires (> 5)"])
        fire_error = test.groupby("fire_bucket")["abs_error"].agg(["mean", "count"])
        for bucket, row in fire_error.iterrows():
            print(f"{str(bucket).ljust(18)}: Error = {row['mean']:.2f} µg/m³ (N={int(row['count'])})")
    else:
        print("fire_density_100km not found in feature set.")

    # 8. Top 5% Worst
    print("\n--- 4. The 'Worst 5%' Autopsy (Catastrophic Failures) ---")
    threshold = test["abs_error"].quantile(0.95)
    worst_5 = test[test["abs_error"] >= threshold]
    normal = test[test["abs_error"] < threshold]
    
    print(f"Threshold for Top 5% Worst Errors: >= {threshold:.2f} µg/m³")
    
    compare_cols = ["abs_error", "value", "predicted", "pm25_lag_1", "pm25_ema_3d", "future_wind", "future_temp", "future_precip"]
    if "fire_density_100km" in normal.columns:
        compare_cols.append("fire_density_100km")
        
    print("\nAverage feature values for Normal vs Worst 5%:")
    
    worst_means = worst_5[compare_cols].mean()
    normal_means = normal[compare_cols].mean()
    
    print(f"{'Feature'.ljust(22)} | {'Normal'.ljust(10)} | {'Worst 5%'.ljust(10)}")
    print("-" * 50)
    for col in compare_cols:
        val_normal = normal_means[col]
        val_worst = worst_means[col]
        print(f"{col.ljust(22)} | {val_normal:<10.2f} | {val_worst:<10.2f}")
        
    print("\nAutopsy Complete.\n")

if __name__ == "__main__":
    main()
