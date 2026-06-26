import psycopg2, sys, os, json
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from src.config import DB_CONFIG

V11_MODEL_DIR = os.path.join("models", "v11")
COUNTRIES = ["IN", "US", "GB", "AU"]

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

def get_actuals(cc):
    cur.execute("""
        SELECT d.date, AVG(d.value) 
        FROM daily_features d
        JOIN stations s ON d.station_id = s.id
        WHERE s.country_code = %s
          AND d.date BETWEEN '2026-06-21' AND '2026-06-25'
          AND d.parameter = 'pm25'
          AND d.value IS NOT NULL
        GROUP BY d.date
        ORDER BY d.date
    """, (cc,))
    return {str(r[0]): r[1] for r in cur.fetchall()}

for cc in COUNTRIES:
    # 1. Load the h1 model to do a simple 1-day prediction test on the dates
    model_path = os.path.join(V11_MODEL_DIR, f"{cc}_pm25_h1_xgb.json")
    meta_path = os.path.join(V11_MODEL_DIR, f"{cc}_pm25_h1_meta.json")
    if not os.path.exists(model_path): continue
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    # 2. Grab healed data for the validation period
    sql = """
        SELECT df.* 
        FROM daily_features df
        JOIN stations s ON df.station_id = s.id
        WHERE s.country_code = %s
          AND df.parameter = 'pm25'
          AND df.value IS NOT NULL
          AND df.date BETWEEN '2026-06-21' AND '2026-06-25'
    """
    df = pd.read_sql(sql, conn, params=(cc,))
    if df.empty: continue
    
    actuals = get_actuals(cc)
    print(f"\n--- {cc} Re-Validation (June 21-25) ---")
    
    # We will compute the average prediction per day to compare against actual
    for date_str, actual_val in actuals.items():
        day_df = df[df['date'].astype(str) == date_str].copy()
        if day_df.empty: continue
        
        # Inject the new weather features into the future_* columns as the model expects
        day_df['future_temp'] = day_df['om_temperature']
        day_df['future_wind'] = day_df['om_wind_speed']
        day_df['future_precip'] = day_df['om_precipitation']
        
        # Ensure all columns exist
        for col in feature_cols:
            if col not in day_df.columns:
                day_df[col] = 0
                
        # Fill NaNs
        X_test = day_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        preds = model.predict(X_test)
        
        # V11 Predicts DELTA (so we add it to the lag_1 to get final prediction)
        final_preds = preds + day_df['lag_1'].fillna(0).values
        avg_pred = np.mean(final_preds)
        
        mae = abs(actual_val - avg_pred)
        print(f"Date: {date_str} | Actual: {actual_val:.2f} | Predicted: {avg_pred:.2f} | MAE: {mae:.2f}")

