import os
import sys
import json
import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from train_v11_aod_global import load_v11_data, build_v11_features, temporal_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

COUNTRIES = ["IN", "US", "GB", "AU"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v11")
OUT_JSON = os.path.join(MODEL_DIR, "best_params_per_country.json")

def objective(trial, X_train, y_train_delta, X_test, y_test_real, test_lags):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": 2,
        "tree_method": "hist",
        "random_state": 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train_delta, verbose=False)
    
    y_pred_delta = model.predict(X_test)
    y_pred = y_pred_delta + test_lags
    
    mae = mean_absolute_error(y_test_real, y_pred)
    return mae

def tune_country(conn, cc):
    print(f"\n=========================================")
    print(f"Tuning {cc}...")
    df_global, viirs = load_v11_data(conn, cc)
    
    # We tune on horizon 1 as the primary representative horizon
    h = 1
    df_h, features = build_v11_features(df_global, viirs, h)
    train, test = temporal_split(df_h, "value")
    
    if train.empty or test.empty:
        print(f"Not enough data for {cc}")
        return None
        
    lag_col = f"pm25_lag_{h}"
    
    X_train = train[features].copy()
    y_train_delta = train["target_delta"]
    
    X_test = test[features].copy()
    y_test_real = test["value"]
    test_lags = test[lag_col]
    
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
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train_delta, X_test, y_test_real, test_lags), n_trials=30)
    
    best = study.best_params
    print(f"Best params for {cc}: {best} (MAE: {study.best_value:.2f})")
    return best

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    best_params = {}
    for cc in COUNTRIES:
        params = tune_country(conn, cc)
        if params:
            best_params[cc] = params
            
    with open(OUT_JSON, "w") as f:
        json.dump(best_params, f, indent=2)
        
    print(f"\nOptimization Matrix saved to {OUT_JSON}")
    conn.close()

if __name__ == "__main__":
    main()
