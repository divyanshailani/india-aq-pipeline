import os
import sys
import time
import json
import xgboost as xgb
import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG
from train_v11_aod_global import load_v11_data, build_v11_features

HORIZONS = [1, 7, 14, 30]
COUNTRIES = ["IN", "GB", "US", "AU"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "v11")

def train_full_v11(df_global, viirs, country_code, horizon, best_all):
    if country_code == "GB" and horizon in [14, 30]:
        print(f"\n  ── {country_code}  h={horizon:>2}d ── SKIPPING (Fallback to V9)")
        return None
        
    print(f"\n  ── {country_code}  h={horizon:>2}d (V11 AOD FULL COMPILE) ──")
    if len(df_global) < 100: return None

    df_h, features = build_v11_features(df_global, viirs, horizon)
    target_col = "value"
    
    # 100% of data (No temporal split)
    train = df_h.copy()
    if train.empty: return None

    X_train = train[features].copy()
    y_train_delta = train["target_delta"]

    medians = {}
    for col in features:
        med = X_train[col].median()
        if hasattr(med, '__len__'):
            med = med.iloc[0] if len(med) > 0 else 0.0
        medians[col] = med if not (isinstance(med, float) and pd.isna(med)) else 0.0
        X_train[col] = X_train[col].fillna(medians[col])

    X_train = X_train.replace([np.inf, -np.inf], 0)

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
    
    if country_code in best_all:
        best_cc = best_all[country_code]
        for k in ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree"]:
            if k in best_cc:
                params[k] = best_cc[k]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train_delta, verbose=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_h{horizon}_xgb.json")
    model.save_model(model_path)

    meta_path = os.path.join(MODEL_DIR, f"{country_code}_pm25_h{horizon}_meta.json")
    
    existing_metrics = {
        "accuracy_percentage": 0.0,
        "test_mae": 0.0,
        "note": "Trained on 100% full dataset for maximum recency."
    }
    
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            old_meta = json.load(f)
            if "metrics" in old_meta and old_meta["metrics"].get("accuracy_percentage", 0) > 0:
                existing_metrics = old_meta["metrics"]
                existing_metrics["note"] = "Trained on 100% full dataset for maximum recency. Metrics carried forward from holdout evaluation."

    meta = {
        "country": country_code,
        "model": "XGBRegressor",
        "version": "v11_xgboost_aod_full",
        "horizon_days": horizon,
        "features": features,
        "feature_medians": medians,
        "metrics": existing_metrics
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

def main():
    start = time.time()
    conn  = psycopg2.connect(**DB_CONFIG)

    param_path = os.path.join(MODEL_DIR, "best_params_per_country.json")
    best_all = {}
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_all = json.load(f)

    for cc in COUNTRIES:
        print(f"\n=========================================")
        print(f"Loading data for {cc}...")
        df_global, viirs = load_v11_data(conn, cc)
        for h in HORIZONS:
            train_full_v11(df_global, viirs, cc, h, best_all)

    conn.close()
    elapsed = time.time() - start
    print(f"\nTotal Full Compile Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
