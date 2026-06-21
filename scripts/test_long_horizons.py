import os
import sys
import glob
import warnings
import xgboost as xgb
import numpy as np
import pandas as pd
import psycopg2
import optuna
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

def load_base_data(conn, country_code):
    sql = """
        SELECT df.station_id, df.date, df.value,
               df.om_temperature as future_temp,
               df.om_wind_speed as future_wind,
               df.om_precipitation as future_precip,
               df.wind_direction,
               s.latitude as station_lat,
               s.longitude as station_lon
        FROM daily_features df
        JOIN stations s ON df.station_id = s.id
        WHERE df.country_code = %s
          AND df.parameter = 'pm25'
          AND df.value IS NOT NULL
        ORDER BY df.station_id, df.date
    """
    df = pd.read_sql(sql, conn, params=(country_code,))
    df["date"] = pd.to_datetime(df["date"])
    
    try:
        csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "DL_FIRE_SV-C2_*", "fire_*.csv"))
        viirs_list = []
        for f in csv_files:
            v = pd.read_csv(f, usecols=["latitude", "longitude", "brightness", "acq_date"])
            v = v[(v["latitude"] >= 6) & (v["latitude"] <= 38) & 
                  (v["longitude"] >= 68) & (v["longitude"] <= 98)]
            v = v.rename(columns={"latitude": "fire_lat", "longitude": "fire_lon"})
            viirs_list.append(v)
            
        if viirs_list:
            viirs = pd.concat(viirs_list, ignore_index=True)
            viirs["acq_date"] = pd.to_datetime(viirs["acq_date"])
        else:
            raise FileNotFoundError("No VIIRS CSV files found in data/raw/")
    except Exception as e:
        print(f"Warning: Could not load viirs_data from CSV. Error: {e}")
        viirs = pd.DataFrame(columns=["fire_lat", "fire_lon", "brightness", "acq_date"])
        
    return df, viirs

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    a = np.sin(dLat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_v9_features(df, viirs, horizon):
    df_v9 = df.copy()
    
    df_v9["fire_density_100km"] = 0.0
    df_v9["fire_radiative_power_total"] = 0.0
    
    if not viirs.empty:
        for date, group in df_v9.groupby("date"):
            viirs_date = viirs[viirs["acq_date"] == date]
            if viirs_date.empty: continue
            for idx, row in group.iterrows():
                dists = haversine_dist(row["station_lat"], row["station_lon"], viirs_date["fire_lat"].values, viirs_date["fire_lon"].values)
                mask = dists <= 100.0
                if np.any(mask):
                    df_v9.at[idx, "fire_density_100km"] = float(np.sum(mask))
                    df_v9.at[idx, "fire_radiative_power_total"] = float(np.sum(viirs_date["brightness"].values[mask]))
                    
    df_v9 = df_v9.sort_values(["station_id", "date"])
    
    lag_col = f"pm25_lag_{horizon}"
    df_v9[lag_col] = df_v9.groupby("station_id")["value"].shift(horizon)
    
    lag_col_next = f"pm25_lag_{horizon+1}"
    df_v9[lag_col_next] = df_v9.groupby("station_id")["value"].shift(horizon+1)
    
    df_v9["month"] = df_v9["date"].dt.month
    df_v9["day_of_year"] = df_v9["date"].dt.dayofyear
    df_v9["day_of_week"] = df_v9["date"].dt.dayofweek
    
    df_v9['pm25_ema_3d'] = df_v9.groupby('station_id')["value"].transform(lambda x: x.ewm(span=3, adjust=False).mean()).shift(horizon)

    df_v9['month_sin'] = np.sin(2 * np.pi * df_v9["month"] / 12)
    df_v9['month_cos'] = np.cos(2 * np.pi * df_v9["month"] / 12)
    df_v9['day_of_year_sin'] = np.sin(2 * np.pi * df_v9["day_of_year"] / 365.25)
    df_v9['day_of_year_cos'] = np.cos(2 * np.pi * df_v9["day_of_year"] / 365.25)
    
    df_v9["pm25_momentum"] = df_v9[lag_col] - df_v9[lag_col_next]
    
    df_v9["future_temp_momentum"] = df_v9.groupby("station_id")["future_temp"].diff(1)
    df_v9["future_wind_momentum"] = df_v9.groupby("station_id")["future_wind"].diff(1)
    
    if "wind_direction" in df_v9.columns:
        wd = df_v9["wind_direction"].fillna(0)
        df_v9["wind_u"] = np.cos(wd * np.pi / 180)
        df_v9["wind_v"] = np.sin(wd * np.pi / 180)
    
    df_v9 = df_v9.dropna(subset=[
        lag_col, lag_col_next, 
        "pm25_ema_3d",
        "future_temp_momentum", "future_wind_momentum"
    ])
    
    df_v9["fire_wind_interaction"] = df_v9["fire_radiative_power_total"] * df_v9["future_wind"]
    
    features = [
        lag_col,
        "pm25_ema_3d",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "future_temp", "future_wind", "future_precip",
        "pm25_momentum", "future_temp_momentum", "future_wind_momentum",
        "fire_density_100km", "fire_radiative_power_total",
        "fire_wind_interaction"
    ]
    if "wind_direction" in df_v9.columns:
        features.extend(["wind_u", "wind_v"])
        
    return df_v9, features

def temporal_split(df, target_col, test_ratio=0.20):
    train_parts, test_parts = [], []
    for sid, group in df.groupby("station_id"):
        group = group.sort_values("date")
        n = len(group)
        split_idx = int(n * (1 - test_ratio))
        if split_idx < 10 or (n - split_idx) < 3:
            continue
        split_date = group.iloc[split_idx]["date"]
        train_mask = group["date"] < split_date
        test_mask  = group["date"] >= split_date
        train_parts.append(group[train_mask])
        test_parts.append(group[test_mask])
    if not train_parts: return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    country_code = "IN"
    horizons = [7, 14, 30]
    
    # Load once to save time
    print("Loading global base data and VIIRS CSVs...")
    df_global, viirs = load_base_data(conn, country_code)
    target_col = "value"
    
    results = {}
    
    for horizon in horizons:
        print(f"\n=========================================")
        print(f"Evaluating Horizon: h={horizon}")
        print(f"=========================================")
        
        df_h, features = build_v9_features(df_global, viirs, horizon)
        train, test = temporal_split(df_h, target_col)
        
        lag_col = f"pm25_lag_{horizon}"
        y_test = test[target_col].copy()
        y_naive = test[lag_col].copy()
        naive_mae = mean_absolute_error(y_test, y_naive)
        print(f"Naive MAE (h={horizon}): {naive_mae:.4f}")

        X_train = train[features].copy()
        y_train = train[target_col].copy()
        X_test  = test[features].copy()

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

        # Delta Target Math
        y_train_delta = train[target_col] - train[lag_col]
        y_test_delta  = test[target_col] - test[lag_col]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": 300,
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "tree_method": "hist",
                "eval_metric": "mae",
                "early_stopping_rounds": 10,
                "random_state": 42
            }

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train_delta, eval_set=[(X_test, y_test_delta)], verbose=False)
            y_pred_delta = model.predict(X_test)
            y_pred_final = y_pred_delta + test[lag_col]
            
            test_mae = mean_absolute_error(y_test, y_pred_final)
            return test_mae

        print(f"Starting Optuna study for h={horizon} (50 trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        best_mae = study.best_value
        best_mase = best_mae / naive_mae if naive_mae > 0 else 0

        print(f"Best MAE:  {best_mae:.4f}")
        print(f"Best MASE: {best_mase:.4f} (Baseline Naive MAE: {naive_mae:.4f})")
        results[horizon] = {
            "naive_mae": naive_mae,
            "best_mae": best_mae,
            "best_mase": best_mase,
            "params": best_params
        }

    print("\n\n=========================================")
    print("FINAL SUMMARY (LONG HORIZONS)")
    print("=========================================")
    for h, res in results.items():
        print(f"h={h:2d} | MASE: {res['best_mase']:.4f} | MAE: {res['best_mae']:.4f} (vs Naive {res['naive_mae']:.4f})")

    conn.close()

if __name__ == "__main__":
    main()
