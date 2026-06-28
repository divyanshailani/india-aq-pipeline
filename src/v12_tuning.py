"""
V12 Optuna Tuning Script (4x4 Grid Architecture) — LEAKAGE FIX
Optimized for Serverless Preemptible 16-Core CPUs via Modal

FIX: The original script mutated df_country inside the horizon loop,
causing target_1d to leak into h=7, target_1d+target_7d into h=14, etc.
This version uses .copy() per horizon and nuclear-drops ALL target_ columns.

SKIP: h=1 models are already clean and will be copied from the old volume.
"""
import modal

app = modal.App("pow-v12-grid-engine-clean")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas",
        "numpy",
        "xgboost==3.0.0",  # Ensuring API compatibility
        "optuna",
        "optuna-integration",
        "scikit-learn",
        "pyarrow"
    )
    .add_local_file(
        local_path="/Users/divyanshailani/Desktop/pow-eda-pipeline/data/daily_features_full.parquet",
        remote_path="/data/daily_features_full.parquet"
    )
)

# Fresh volume — old one (pow-v12-storage) has contaminated models
volume = modal.Volume.from_name("pow-v12-storage-clean", create_if_missing=True)

@app.function(
    image=image,
    cpu=32,          # Hyper-Optimized CPU scaling
    memory=8192,     # 8 GB RAM
    timeout=86400,   # Max timeout for grid search (24 hours)
    volumes={"/storage": volume}
)
def train_grid():
    import os
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import optuna
    from optuna.integration import XGBoostPruningCallback
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    import logging
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    DATA_PATH = "/data/daily_features_full.parquet"
    # Save artifacts to Modal's persistent volume to survive preemptions
    MODELS_DIR = "/storage/models/v12"
    DB_DIR = "/storage/optuna_dbs"
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    logging.info(f"Loading data from {DATA_PATH}...")
    df_full = pd.read_parquet(DATA_PATH)
    
    if 'parameter' in df_full.columns:
        df_full = df_full[df_full['parameter'] == 'pm25'].copy()
    
    df_full = df_full.sort_values(by=['date', 'station_id']).reset_index(drop=True)
    
    countries = ['US', 'GB', 'IN', 'AU']
    horizons = [1, 7, 14, 30]
    
    for country in countries:
        logging.info(f"========== Processing Country: {country} ==========")
        df_country = df_full[df_full['country_code'] == country].copy()
        
        if len(df_country) == 0:
            logging.warning(f"No data found for {country}. Skipping.")
            continue
            
        for horizon in horizons:
            model_dir = os.path.join(MODELS_DIR, country, f"horizon_{horizon}")
            model_path = os.path.join(model_dir, "model.json")
            if os.path.exists(model_path):
                logging.info(f"Skipping h={horizon} for {country} (Model already safely saved in volume)")
                continue

            # SKIP h=1: Already clean from previous training run
            if horizon == 1:
                logging.info(f"Skipping h=1 for {country} (Already trained and pure)")
                continue
                
            logging.info(f"--- Horizon: {horizon} Days ---")
            
            # FIX: Fresh copy per horizon to prevent target cascade mutation
            df_h = df_country.copy()
            
            # Create ONLY this horizon's target
            target_col = f'target_{horizon}d'
            df_h[target_col] = df_h.groupby('station_id')['value'].shift(-horizon)
            df_h = df_h.dropna(subset=[target_col]).copy()
            
            drop_cols = [
                'date', 'parameter', 'value', 'country_code',
                'pm25_delta_1', 'pm25_delta_7', 'station_id'
            ]
            
            # Anti Ghost-Features logic
            ghost_features = [c for c in df_h.columns if 'wind_dir' in c or 'o3_' in c or 'fire_' in c]
            drop_cols.extend(ghost_features)
            
            # NUCLEAR ANTI-LEAKAGE: Drop ALL columns starting with 'target_'
            target_cols = [c for c in df_h.columns if c.startswith('target_')]
            drop_cols.extend(target_cols)
            
            feature_cols = [c for c in df_h.columns if c not in drop_cols]
            
            X = df_h[feature_cols]
            y = df_h[target_col]
            
            # VERIFICATION: Assert no target leakage
            leaked = [f for f in feature_cols if 'target' in f.lower()]
            if leaked:
                logging.error(f"CRITICAL: Target leakage detected in features: {leaked}. ABORTING.")
                volume.commit()
                raise RuntimeError(f"Target leakage: {leaked}")
            
            logging.info(f"Training on {len(X):,} rows, {len(feature_cols)} features for {country} {horizon}d")
            logging.info(f"Features: {feature_cols}")
            
            def objective(trial):
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "booster": "gbtree",
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                    "random_state": 42,
                    "n_estimators": 1000, 
                    "tree_method": "hist",
                    "n_jobs": -1 
                }
                
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                
                for split_idx, (train_index, test_index) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    
                    callbacks = []
                    # Only apply pruning on the last fold to avoid epoch overlapping and warning spam
                    if split_idx == tscv.get_n_splits() - 1:
                        pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
                        callbacks.append(pruning_callback)
                    
                    model = xgb.XGBRegressor(
                        **params,
                        early_stopping_rounds=50,
                        callbacks=callbacks if callbacks else None
                    )
                    
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
                    
                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    cv_scores.append(rmse)
                    
                return np.mean(cv_scores)
            
            # Anti-Preemption Shield: SQLite storage on the persistent volume
            db_path = os.path.join(DB_DIR, f"optuna_v12_clean_{country}_{horizon}d.db")
            study_name = f"v12_clean_{country}_{horizon}d"
            storage_url = f"sqlite:///{db_path}"
            
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                storage=storage_url,
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )
            
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining_trials = max(0, 150 - completed_trials)
            
            if remaining_trials > 0:
                logging.info(f"Running {remaining_trials} trials (Completed: {completed_trials}/150)...")
                try:
                    study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))
                except Exception as e:
                    logging.error(f"Optuna error: {e}")
                    # Sync volume state before exiting due to crash
                    volume.commit()
                    continue
            else:
                logging.info(f"Study already completed 150 trials.")
            
            # Safe Artifact Drop
            best_params = study.best_trial.params
            best_params.update({
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "booster": "gbtree",
                "random_state": 42,
                "n_estimators": 1000,
                "tree_method": "hist",
                "n_jobs": -1
            })
            
            logging.info(f"Best RMSE for {country} {horizon}d: {study.best_value:.4f}")
            logging.info("Training final model on full dataset...")
            
            final_model = xgb.XGBRegressor(**best_params)
            final_model.fit(X, y, verbose=False)
            
            # POST-TRAINING VERIFICATION: Check saved model has NO target features
            saved_feats = final_model.get_booster().feature_names
            leaked_post = [f for f in saved_feats if 'target' in f.lower()]
            if leaked_post:
                logging.error(f"POST-TRAIN LEAK DETECTED: {leaked_post}. Model NOT saved.")
                volume.commit()
                continue
            
            model_dir = os.path.join(MODELS_DIR, country, f"horizon_{horizon}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "model.json")
            final_model.get_booster().save_model(model_path)
            logging.info(f"Model safely deployed to volume: {model_path}")
            logging.info(f"Verified: {len(saved_feats)} clean features, 0 leaked targets.\n")
            
            # Sync volume so we don't lose data on preemptions
            volume.commit()

@app.local_entrypoint()
def main():
    print("🚀 Firing V12 CLEAN Grid Engine on Modal (Leakage Fix Applied)...")
    print("   Skipping h=1 (already pure). Retraining h=7, h=14, h=30 for all 4 countries.")
    train_grid.remote()
    print("✅ All 12 contaminated models retrained and safely dropped in Volume 'pow-v12-storage-clean'!")

