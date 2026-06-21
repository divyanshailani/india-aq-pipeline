import os
import sys
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


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    country_code = "IN"
    horizon = 1
    
    print(f"Loading data for {country_code} h={horizon}...")
    df = load_base_data(conn, country_code)
    df_h, features = build_v9_features(df, horizon)
    target_col = "value"
    train, test = temporal_split(df_h, target_col)
    
    y_test = test[target_col].copy()
    y_naive = test[f"pm25_lag_{horizon}"].copy()
    naive_mae = mean_absolute_error(y_test, y_naive)
    print(f"Naive MAE: {naive_mae:.4f}")

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

    def objective(trial):
        params = {
            "n_estimators": 300,
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method": "hist",
            "eval_metric": "mae",
            "early_stopping_rounds": 10,
            "random_state": 42
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred_test = model.predict(X_test)
        
        test_mae = mean_absolute_error(y_test, y_pred_test)
        return test_mae

    print("Starting Optuna study...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n===============================")
    print("Optimization Complete")
    print("===============================")
    best_params = study.best_params
    best_mae = study.best_value
    best_mase = best_mae / naive_mae if naive_mae > 0 else 0

    print("Best Parameters:")
    print(best_params)
    print(f"Best MAE:  {best_mae:.4f}")
    print(f"Best MASE: {best_mase:.4f} (Baseline Naive MAE: {naive_mae:.4f})")
    
    conn.close()

if __name__ == "__main__":
    main()
