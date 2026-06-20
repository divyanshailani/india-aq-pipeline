"""
Global Air Quality Model — Retrain Pipeline (V9 XGBoost)
============================================================
This script is triggered by the Admin Dashboard's "Continuous Retraining" button.
It runs the V9 XGBoost engine training script and then automatically
kicks off the predict pipeline to sync the new models to the live site.
"""

import os
import sys
import subprocess
import time

def main():
    print("═" * 62)
    print("  GLOBAL AQ MODEL — RETRAIN PIPELINE (V9 XGBOOST)")
    print("═" * 62)
    
    start = time.time()
    scripts_dir = os.path.dirname(__file__)
    
    # 1. Train V9 Models
    train_script = os.path.join(scripts_dir, "train_v9_xgboost.py")
    print("\n[1/2] Triggering V9 Model Training...")
    try:
        subprocess.run([sys.executable, train_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"🚨 Training failed: {e}")
        sys.exit(1)
        
    # 2. Ripple Effect: Predict Pipeline
    print(f"\n{'═'*62}")
    print(f"  [2/2] TRIGGERING RIPPLE EFFECT (Predict Pipeline)")
    print(f"{'═'*62}")
    predict_script = os.path.join(scripts_dir, "predict_pipeline.py")
    try:
        subprocess.run([sys.executable, predict_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"🚨 Prediction failed: {e}")
        sys.exit(1)
        
    elapsed = time.time() - start
    print(f"\n{'═'*62}")
    print(f"  COMPLETE  ({int(elapsed)}s)")
    print(f"{'═'*62}")

if __name__ == "__main__":
    main()
