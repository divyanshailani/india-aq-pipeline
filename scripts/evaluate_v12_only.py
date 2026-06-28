import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation import calculate_mae, calculate_nmae, calculate_mase, calculate_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

V12_MODEL_DIR = os.path.join("models", "v12")
PREDICTIONS_DIR = os.path.join("data", "predictions_v12")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

HOLDOUT_START = "2026-01-01"

def main():
    logging.info("Starting V12 (Challenger) Evaluation...")
    
    logging.info("Loading V12 Parquet data...")
    df_v12_base = pd.read_parquet("data/daily_features_full.parquet")
    
    countries = ["US", "GB", "IN", "AU"]
    horizons = [1, 7, 14, 30]
    
    results = []
    
    for country in countries:
        logging.info(f"========== {country} ==========")
        df_v12_country = df_v12_base[df_v12_base['country_code'] == country].copy()
        df_v12_country['date'] = pd.to_datetime(df_v12_country['date'])
        
        for horizon in horizons:
            logging.info(f"--- Horizon {horizon}d ---")
            
            # Prepare Target Shift for V12
            df_h = df_v12_country.copy().sort_values(['station_id', 'date'])
            target_col = f'target_{horizon}d'
            for h_temp in [1, 7, 14, 30]:
                df_h[f'target_{h_temp}d'] = df_h.groupby('station_id')['value'].shift(-h_temp)
                
            df_h['target_date'] = df_h['date'] + pd.Timedelta(days=horizon)
            
            # Drop rows where target is NaN (we can't evaluate without true future target)
            df_h = df_h.dropna(subset=[target_col])
            
            # Filter for Holdout Period
            # We want predictions made ON OR AFTER Jan 1, 2026
            df_test = df_h[df_h['date'] >= pd.to_datetime(HOLDOUT_START)].copy()
            
            if len(df_test) == 0:
                logging.warning(f"No holdout data for {country} h={horizon}")
                continue
                
            v12_model_dir = os.path.join(V12_MODEL_DIR, country, f"horizon_{horizon}")
            if not os.path.exists(v12_model_dir):
                logging.warning(f"V12 model missing for {country} h={horizon}. Skipping.")
                continue
                
            v12_model = xgb.XGBRegressor()
            v12_model.load_model(os.path.join(v12_model_dir, "model.json"))
            
            v12_features = ['month', 'day_of_week', 'is_weekend', 'day_of_year',
                            'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
                            'roll_3_mean', 'roll_7_mean', 'roll_3_std', 'roll_14_mean',
                            'roll_30_mean', 'roll_14_std', 'om_temperature', 'om_wind_speed',
                            'om_precipitation', 'om_aerosol_optical_depth',
                            'rolling_3day_precip', 'aod_volatility_index',
                            'latitude', 'longitude']
            
            # Wait, earlier we found target_1d was leaked into V12 features for horizon 7!
            # If the model expects it, it will crash if we don't provide it.
            # We can use model.get_booster().feature_names to get the EXACT list of features the model was trained with.
            v12_features = v12_model.get_booster().feature_names
            # Drop rows missing required features for V12 inference
            df_test = df_test.dropna(subset=v12_features).copy()
            
            if len(df_test) == 0:
                logging.warning(f"No test data left after dropping NA features for {country} h={horizon}")
                continue
                
            X_v12 = df_test[v12_features]
            v12_preds = v12_model.predict(X_v12)
            
            df_test['v12_pred'] = v12_preds
            
            # True actual is the phase-shifted future value
            y_true = df_test[target_col]
            y_pred = df_test['v12_pred']
            # For MASE, the naive baseline is just predicting today's value for the future
            y_naive = df_test['value']
            
            mae = calculate_mae(y_true, y_pred)
            nmae = calculate_nmae(y_true, y_pred)
            mase = calculate_mase(y_true, y_pred, y_naive)
            acc = calculate_accuracy(nmae)
            
            res = {
                "Country": country,
                "Horizon": f"{horizon}d",
                "Samples": len(df_test),
                "MAE": f"{mae:.2f}",
                "NMAE": f"{nmae:.3f}",
                "MASE": f"{mase:.3f}",
                "Accuracy": f"{acc:.1f}%"
            }
            results.append(res)
            
            # Save predictions
            out_file = os.path.join(PREDICTIONS_DIR, f"eval_v12_{country}_{horizon}.csv")
            # We save date, station_id, current_value, target_date, true_target, prediction
            df_out = df_test[['date', 'target_date', 'station_id', 'value', target_col, 'v12_pred']].rename(
                columns={'value': 'current_pm25', target_col: 'actual_future_pm25'}
            )
            df_out.to_csv(out_file, index=False)
            logging.info(f"Saved {len(df_test)} predictions to {out_file}")
            
    # Print Markdown Table
    print("\n# V12 (Challenger) First-Principles Evaluation")
    print("**Holdout Period:** 2026-01-01 to Latest")
    print("| Country | Horizon | Samples | MAE | NMAE | MASE | Accuracy |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Country']} | {r['Horizon']} | {r['Samples']} | {r['MAE']} | {r['NMAE']} | {r['MASE']} | {r['Accuracy']} |")

if __name__ == "__main__":
    main()
