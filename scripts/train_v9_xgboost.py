import os
import sys
import time
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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v9")

def load_base_data(conn, country_code):
    sql = """
        SELECT station_id, date, value,
               om_temperature as future_temp,
               om_wind_speed as future_wind,
               om_precipitation as future_precip
        FROM daily_features
        WHERE country_code = %s
          AND parameter = 'pm25'
          AND value IS NOT NULL
        ORDER BY station_id, date
    """
    df = pd.read_sql(sql, conn, params=(country_code,))
    df["date"] = pd.to_datetime(df["date"])
    return df

def build_v9_features(df, horizon):
    df_v9 = df.copy()
    df_v9 = df_v9.sort_values(["station_id", "date"])
    
    lag_col = f"pm25_lag_{horizon}"
    df_v9[lag_col] = df_v9.groupby("station_id")["value"].shift(horizon)
    
    roll_mean = df_v9.groupby("station_id")["value"].rolling(3).mean().reset_index(level=0, drop=True)
    df_v9["pm25_rolling_mean_3d"] = roll_mean.groupby(df_v9["station_id"]).shift(horizon)
    
    roll_std = df_v9.groupby("station_id")["value"].rolling(3).std().reset_index(level=0, drop=True)
    df_v9["pm25_rolling_std_3d"] = roll_std.groupby(df_v9["station_id"]).shift(horizon)
    
    df_v9["month"] = df_v9["date"].dt.month
    df_v9["day_of_year"] = df_v9["date"].dt.dayofyear
    df_v9["day_of_week"] = df_v9["date"].dt.dayofweek
    
    df_v9 = df_v9.dropna(subset=[lag_col, "pm25_rolling_mean_3d", "pm25_rolling_std_3d"])
    
    features = [
        lag_col, "pm25_rolling_mean_3d", "pm25_rolling_std_3d",
        "month", "day_of_year", "day_of_week",
        "future_temp", "future_wind", "future_precip"
    ]
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

def train_horizon_model(conn, country_code, horizon):
    print(f"\n  ── {country_code}  h={horizon:>2}d ──")
    df = load_base_data(conn, country_code)
    if len(df) < 100: return None

    df_h, features = build_v9_features(df, horizon)
    target_col = "value"
    train, test = temporal_split(df_h, target_col)
    if train.empty or test.empty: return None

    y_test = test[target_col].copy()
    y_naive = test[f"pm25_lag_{horizon}"].copy()
    naive_mae = mean_absolute_error(y_test, y_naive)

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

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, 
        subsample=0.8, eval_metric="mae", early_stopping_rounds=10, 
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    
    train_r2  = r2_score(y_train, y_pred_train)
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
        "version": "v9_xgboost_global",
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
        all_meta[cc] = {}
        for h in HORIZONS:
            meta = train_horizon_model(conn, cc, h)
            if meta:
                all_meta[cc][f"h{h}"] = meta

    conn.close()
    
    print("\n### V9 Global XGBoost Engine - Final Evaluation")
    print("| Country | Horizon | MAE | NMAE | MASE | Accuracy (%) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for cc in COUNTRIES:
        for h in HORIZONS:
            m = all_meta.get(cc, {}).get(f"h{h}", {})
            if m:
                print(f"| {cc} | {h} | {m['metrics']['test_mae']:.2f} | {m['metrics']['nmae']:.4f} | {m['metrics']['mase']:.4f} | {m['metrics']['accuracy_percentage']:.2f}% |")

if __name__ == "__main__":
    main()
