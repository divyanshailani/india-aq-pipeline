import psycopg2, sys, os, json
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from src.config import DB_CONFIG

V11_MODEL_DIR = os.path.join("models", "v11")
COUNTRIES = ["IN", "US", "GB", "AU"]

conn = psycopg2.connect(**DB_CONFIG)

for cc in COUNTRIES:
    model_path = os.path.join(V11_MODEL_DIR, f"{cc}_pm25_h1_xgb.json")
    meta_path = os.path.join(V11_MODEL_DIR, f"{cc}_pm25_h1_meta.json")
    if not os.path.exists(model_path): continue
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    sql = """
        SELECT df.* 
        FROM daily_features df
        JOIN stations s ON df.station_id = s.id
        WHERE s.country_code = %s
          AND df.parameter = 'pm25'
          AND df.value IS NOT NULL
          AND df.lag_1 IS NOT NULL
          AND df.date BETWEEN '2026-06-21' AND '2026-06-25'
    """
    df = pd.read_sql(sql, conn, params=(cc,))
    if df.empty: continue
    
    # We do NOT inject healed weather here, so it acts blind (temp=0, precip=0)
    # Actually wait, om_temperature is what is in DB now. We must zero it out to simulate "blind"!
    df['future_temp'] = 0
    df['future_wind'] = 0
    df['future_precip'] = 0
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X_test = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    preds = model.predict(X_test)
    y_pred = preds + df['lag_1'].values
    y_actual = df['value'].values
    y_naive = df['lag_1'].values
    
    mae = float(np.mean(np.abs(y_actual - y_pred)))
    mean_y = float(np.mean(y_actual))
    nmae = mae / mean_y if mean_y > 0 else 0
    acc_pct = max(0.0, (1.0 - nmae) * 100.0)
    
    naive_mae = float(np.mean(np.abs(y_actual - y_naive)))
    mase = mae / naive_mae if naive_mae > 0 else 0
    
    print(f"BLIND {cc} -> MAE: {mae:.2f}, NMAE: {nmae:.2f}, Acc: {acc_pct:.1f}%, MASE: {mase:.2f}")

