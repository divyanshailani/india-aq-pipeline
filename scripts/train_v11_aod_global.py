import os
import sys
import time
import glob
import warnings
import json
import xgboost as xgb
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG

HORIZONS = [1, 7, 14, 30]
COUNTRIES = ["IN", "GB", "US", "AU"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v11")

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    a = np.sin(dLat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

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
        LEFT JOIN satellite_aod_features aod 
            ON df.station_id = aod.station_id AND df.date = aod.date
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
    
    lag_col = f"pm25_lag_{horizon}"
    df_v11[lag_col] = df_v11.groupby("station_id")["value"].shift(horizon)
    lag_col_next = f"pm25_lag_{horizon+1}"
    df_v11[lag_col_next] = df_v11.groupby("station_id")["value"].shift(horizon+1)
    
    df_v11["aod_mean_lag_1"] = df_v11.groupby("station_id")["aod_mean"].shift(1)
    df_v11["aod_max_lag_1"] = df_v11.groupby("station_id")["aod_max"].shift(1)
    
    df_v11["aod_mean_lag_1"] = df_v11.groupby("station_id")["aod_mean_lag_1"].transform(lambda x: x.fillna(x.median()))
    df_v11["aod_max_lag_1"] = df_v11.groupby("station_id")["aod_max_lag_1"].transform(lambda x: x.fillna(x.median()))
    df_v11["aod_mean_lag_1"] = df_v11["aod_mean_lag_1"].fillna(df_v11["aod_mean_lag_1"].median())
    df_v11["aod_max_lag_1"] = df_v11["aod_max_lag_1"].fillna(df_v11["aod_max_lag_1"].median())
    
    df_v11["pm25_ema_3d"] = df_v11.groupby("station_id")["value"].transform(lambda x: x.ewm(span=3, adjust=False).mean()).shift(horizon)
    
    df_v11["month"] = df_v11["date"].dt.month
    df_v11["day_of_year"] = df_v11["date"].dt.dayofyear
    df_v11['month_sin'] = np.sin(2 * np.pi * df_v11["month"] / 12)
    df_v11['month_cos'] = np.cos(2 * np.pi * df_v11["month"] / 12)
    df_v11['day_of_year_sin'] = np.sin(2 * np.pi * df_v11["day_of_year"] / 365.25)
    df_v11['day_of_year_cos'] = np.cos(2 * np.pi * df_v11["day_of_year"] / 365.25)
    
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
    
    df_v11["fire_wind_interaction"] = df_v11["fire_radiative_power_total"] * df_v11["future_wind"]
    df_v11["target_delta"] = df_v11["value"] - df_v11[lag_col]
    
    features = [
        lag_col, "pm25_ema_3d",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "future_temp", "future_wind", "future_precip",
        "pm25_momentum", "future_temp_momentum", "future_wind_momentum",
        "fire_density_100km", "fire_radiative_power_total",
        "fire_wind_interaction",
        "aod_mean_lag_1", "aod_max_lag_1"
    ]
    if "wind_direction" in df_v11.columns:
        features.extend(["wind_u", "wind_v"])
        
    return df_v11, features

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

def train_v11_horizon_model(df_global, viirs, country_code, horizon):
    if country_code == "GB" and horizon in [14, 30]:
        print(f"\n  ── {country_code}  h={horizon:>2}d ── SKIPPING (Fallback to V9)")
        return None
        
    print(f"\n  ── {country_code}  h={horizon:>2}d (V11 AOD) ──")
    if len(df_global) < 100: return None

    df_h, features = build_v11_features(df_global, viirs, horizon)
    target_col = "value"
    train, test = temporal_split(df_h, target_col)
    if train.empty or test.empty: return None

    lag_col = f"pm25_lag_{horizon}"
    y_test = test[target_col].copy()
    y_naive = test[lag_col].copy()
    naive_mae = mean_absolute_error(y_test, y_naive)

    X_train = train[features].copy()
    y_train_delta = train["target_delta"]
    
    X_test  = test[features].copy()
    y_test_delta = test["target_delta"]

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
    model.fit(X_train, y_train_delta, eval_set=[(X_test, y_test_delta)], verbose=False)
    
    y_pred_delta_train = model.predict(X_train)
    y_pred_train = y_pred_delta_train + train[lag_col]
    
    y_pred_delta_test = model.predict(X_test)
    y_pred_test = y_pred_delta_test + test[lag_col]
    
    y_train_real = train[target_col]
    
    train_r2  = r2_score(y_train_real, y_pred_train)
    test_r2   = r2_score(y_test, y_pred_test)
    test_mae  = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    mean_y = np.mean(y_test)
    nmae = test_mae / mean_y if mean_y > 0 else 0
    mase = test_mae / naive_mae if naive_mae > 0 else 0
    accuracy_percentage = max(0.0, (1.0 - nmae) * 100.0)

    print(f"     test R²={test_r2:.4f}  MAE={test_mae:.2f}  NMAE={nmae:.4f}  Naive_MAE={naive_mae:.2f}  MASE={mase:.2f}  Acc={accuracy_percentage:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_h{horizon}_xgb.json")
    model.save_model(model_path)

    meta = {
        "country": country_code,
        "model": "XGBRegressor",
        "version": "v11_xgboost_aod",
        "horizon_days": horizon,
        "features": features,
        "feature_medians": medians,
        "metrics": {
            "test_r2": round(test_r2, 4),
            "test_mae": round(test_mae, 2),
            "test_rmse": round(test_rmse, 2),
            "nmae": round(nmae, 4),
            "mase": round(mase, 2),
            "accuracy_percentage": round(accuracy_percentage, 2),
        }
    }
    meta_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_h{horizon}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

def main():
    start = time.time()
    conn  = psycopg2.connect(**DB_CONFIG)

    all_meta = {}
    for cc in COUNTRIES:
        print(f"\n=========================================")
        print(f"Loading data for {cc}...")
        df_global, viirs = load_v11_data(conn, cc)
        
        all_meta[cc] = {}
        for h in HORIZONS:
            meta = train_v11_horizon_model(df_global, viirs, cc, h)
            if meta:
                all_meta[cc][f"h{h}"] = meta

    conn.close()
    
    print("\n### V11 Global Engine - Final Evaluation")
    print("| Country | Horizon | MAE | NMAE | MASE | Accuracy (%) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for cc in COUNTRIES:
        for h in HORIZONS:
            m = all_meta.get(cc, {}).get(f"h{h}", {})
            if m:
                print(f"| {cc} | {h} | {m['metrics']['test_mae']:.2f} | {m['metrics']['nmae']:.4f} | {m['metrics']['mase']:.4f} | {m['metrics']['accuracy_percentage']:.2f}% |")

    elapsed = time.time() - start
    print(f"\nTotal Pipeline Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
