import os
import sys
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from scripts.train_v9_4_xgboost import temporal_split, haversine_dist

def load_v11_data(conn, country_code):
    sql = """
        SELECT df.station_id, df.date, df.value,
               df.om_temperature as future_temp,
               df.om_wind_speed as future_wind,
               df.om_precipitation as future_precip,
               df.wind_direction,
               s.latitude as station_lat,
               s.longitude as station_lon,
               aod.aod_mean,
               aod.aod_max
        FROM daily_features df
        JOIN stations s ON df.station_id = s.id
        INNER JOIN satellite_aod_features aod 
            ON df.station_id = aod.station_id AND df.date = aod.date
        WHERE df.country_code = %s
          AND df.parameter = 'pm25'
          AND df.value IS NOT NULL
        ORDER BY df.station_id, df.date
    """
    df = pd.read_sql(sql, conn, params=(country_code,))
    df["date"] = pd.to_datetime(df["date"])
    
    import glob
    try:
        csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "DL_FIRE_SV-C2_*", "fire_*.csv"))
        viirs_list = []
        bounds = {"IN": (6, 38, 68, 98)}
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

def build_v11_features(df, viirs, horizon):
    df_v11 = df.copy()
    
    df_v11["fire_density_100km"] = 0.0
    df_v11["fire_radiative_power_total"] = 0.0
    
    if not viirs.empty:
        for date, group in df_v11.groupby("date"):
            viirs_date = viirs[viirs["acq_date"] == date]
            if viirs_date.empty: continue
            for idx, row in group.iterrows():
                dists = haversine_dist(row["station_lat"], row["station_lon"], viirs_date["fire_lat"].values, viirs_date["fire_lon"].values)
                mask = dists <= 100.0
                if np.any(mask):
                    df_v11.at[idx, "fire_density_100km"] = float(np.sum(mask))
                    df_v11.at[idx, "fire_radiative_power_total"] = float(np.sum(viirs_date["brightness"].values[mask]))
                    
    df_v11 = df_v11.sort_values(["station_id", "date"])
    
    # Delta Targets & Lags
    lag_col = f"pm25_lag_{horizon}"
    df_v11[lag_col] = df_v11.groupby("station_id")["value"].shift(horizon)
    lag_col_next = f"pm25_lag_{horizon+1}"
    df_v11[lag_col_next] = df_v11.groupby("station_id")["value"].shift(horizon+1)
    
    # V11 3D Spatial AOD Lags
    df_v11["aod_mean_lag_1"] = df_v11.groupby("station_id")["aod_mean"].shift(1)
    df_v11["aod_max_lag_1"] = df_v11.groupby("station_id")["aod_max"].shift(1)
    
    # Fill NaN for AOD Lags with station medians
    df_v11["aod_mean_lag_1"] = df_v11.groupby("station_id")["aod_mean_lag_1"].transform(lambda x: x.fillna(x.median()))
    df_v11["aod_max_lag_1"] = df_v11.groupby("station_id")["aod_max_lag_1"].transform(lambda x: x.fillna(x.median()))
    df_v11["aod_mean_lag_1"] = df_v11["aod_mean_lag_1"].fillna(df_v11["aod_mean_lag_1"].median())
    df_v11["aod_max_lag_1"] = df_v11["aod_max_lag_1"].fillna(df_v11["aod_max_lag_1"].median())
    
    # Synthetic Memory
    df_v11["pm25_ema_3d"] = df_v11.groupby("station_id")["value"].transform(lambda x: x.ewm(span=3, adjust=False).mean()).shift(horizon)
    
    # Cyclic Time
    df_v11["month"] = df_v11["date"].dt.month
    df_v11["day_of_year"] = df_v11["date"].dt.dayofyear
    df_v11['month_sin'] = np.sin(2 * np.pi * df_v11["month"] / 12)
    df_v11['month_cos'] = np.cos(2 * np.pi * df_v11["month"] / 12)
    df_v11['day_of_year_sin'] = np.sin(2 * np.pi * df_v11["day_of_year"] / 365.25)
    df_v11['day_of_year_cos'] = np.cos(2 * np.pi * df_v11["day_of_year"] / 365.25)
    
    # Micro-Physics Momentum
    df_v11["pm25_momentum"] = df_v11[lag_col] - df_v11[lag_col_next]
    df_v11["future_temp_momentum"] = df_v11.groupby("station_id")["future_temp"].diff(1)
    df_v11["future_wind_momentum"] = df_v11.groupby("station_id")["future_wind"].diff(1)
    
    if "wind_direction" in df_v11.columns:
        wd = df_v11["wind_direction"].fillna(0)
        df_v11["wind_u"] = np.cos(wd * np.pi / 180)
        df_v11["wind_v"] = np.sin(wd * np.pi / 180)
        
    df_v11 = df_v11.dropna(subset=[
        lag_col, lag_col_next, "pm25_ema_3d",
        "future_temp_momentum", "future_wind_momentum"
    ])
    
    # Interaction Term
    df_v11["fire_wind_interaction"] = df_v11["fire_radiative_power_total"] * df_v11["future_wind"]
    
    df_v11["target_delta"] = df_v11["value"] - df_v11[lag_col]
    
    features = [
        lag_col, "pm25_ema_3d",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "future_temp", "future_wind", "future_precip",
        "pm25_momentum", "future_temp_momentum", "future_wind_momentum",
        "fire_density_100km", "fire_radiative_power_total",
        "fire_wind_interaction",
        "aod_mean_lag_1", "aod_max_lag_1"  # V11 AOD Injection
    ]
    if "wind_direction" in df_v11.columns:
        features.extend(["wind_u", "wind_v"])
        
    return df_v11, features

def main():
    print("Initializing V11 AOD Fusion Engine for IN (h=1)...")
    conn = psycopg2.connect(**DB_CONFIG)
    df_in, viirs_data = load_v11_data(conn, "IN")
    conn.close()
    
    print(f"Loaded {len(df_in)} baseline records merged with AOD features.")
    
    df_h, features = build_v11_features(df_in, viirs_data, 1)
    train, test = temporal_split(df_h, "value", test_ratio=0.20)
    
    X_train, y_train = train[features], train["target_delta"]
    X_test, y_test_delta = test[features], test["target_delta"]
    
    y_test_true = test["value"]
    y_test_lag = test["pm25_lag_1"]
    y_train_true = train["value"]
    y_train_lag = train["pm25_lag_1"]
    
    print(f"Training V11 Model with features: {features}")
    
    # Optuna-Optimized Parameters
    params = {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.033,
        "subsample": 0.812,
        "colsample_bytree": 0.502,
        "min_child_weight": 2,
        "tree_method": "hist",
        "random_state": 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test_delta)], verbose=False)
    
    test_pred_delta = model.predict(X_test)
    y_pred = np.maximum(0, test_pred_delta + y_test_lag.values)
    
    mae = mean_absolute_error(y_test_true, y_pred)
    
    naive_err = mean_absolute_error(y_train_true, y_train_lag)
    mase = mae / naive_err if naive_err > 0 else 0
    
    test_df = test.copy()
    test_df["predicted"] = y_pred
    test_df["abs_error"] = np.abs(test_df["value"] - test_df["predicted"])
    
    print("\n" + "="*50)
    print(f"V11 Overall Mean Absolute Error: {mae:.2f} µg/m³")
    print(f"V11 Overall MASE: {mase:.4f}")
    print("="*50)
    
    print("\n--- Extreme Spike Magnitude Slice (True PM2.5 > 150) ---")
    extreme_mask = test_df["value"] > 150
    extreme_test = test_df[extreme_mask]
    
    if not extreme_test.empty:
        extreme_mae = extreme_test["abs_error"].mean()
        print(f"V11 MAE on Extreme Spikes (>150): {extreme_mae:.2f} µg/m³ (N={len(extreme_test)})")
        print(f"(For reference, V9.4 MAE on this slice was 87.48 µg/m³)")
    else:
        print("No extreme spikes found in the test set.")
        
    print("\nSimulation Complete.")

if __name__ == "__main__":
    main()
