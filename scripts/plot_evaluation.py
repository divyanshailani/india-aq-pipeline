import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PREDICTIONS_DIR = os.path.join("data", "predictions")
ARTIFACTS_DIR = "/Users/divyanshailani/.gemini/antigravity/brain/f18c2b27-d05c-474b-8f60-58bdc6cc3c31"

def find_volatile_station(df):
    """Finds the station with the highest variance in actual PM2.5 to test physics."""
    variances = df.groupby('station_id')['actual'].var()
    return variances.idxmax()

def plot_overlay(country="US", horizon=1):
    file_path = os.path.join(PREDICTIONS_DIR, f"holdout_eval_{country}_{horizon}.csv")
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Find volatile station
    target_station = find_volatile_station(df)
    print(f"Overlay Graph: Selected Volatile Station {target_station} for {country}")
    
    df_station = df[df['station_id'] == target_station].sort_values('date')
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_station['date'], df_station['actual'], label='Actual PM2.5', color='blue', linewidth=2, alpha=0.7)
    plt.plot(df_station['date'], df_station['v11_pred'], label='V11.1 (Champion)', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.plot(df_station['date'], df_station['v12_pred'], label='V12 (Challenger)', color='green', linestyle='-', linewidth=2, alpha=0.9)
    
    plt.title(f"Champion vs Challenger Overlay\n{country} - Station: {target_station} - {horizon}d Forecast", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("PM2.5 (µg/m³)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(ARTIFACTS_DIR, f"overlay_graph_{country}_{horizon}d.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def plot_error_decay(country="IN"):
    horizons = [1, 7, 14, 30]
    v11_maes = []
    v12_maes = []
    
    valid_horizons = []
    for h in horizons:
        file_path = os.path.join(PREDICTIONS_DIR, f"holdout_eval_{country}_{h}.csv")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        v11_mae = np.mean(np.abs(df['actual'] - df['v11_pred']))
        v12_mae = np.mean(np.abs(df['actual'] - df['v12_pred']))
        
        valid_horizons.append(h)
        v11_maes.append(v11_mae)
        v12_maes.append(v12_mae)
        
    if not valid_horizons:
        print(f"No data to plot error decay for {country}")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(valid_horizons, v11_maes, marker='o', label='V11.1 (Champion)', color='red', linestyle='--', linewidth=2)
    plt.plot(valid_horizons, v12_maes, marker='s', label='V12 (Challenger)', color='green', linestyle='-', linewidth=2)
    
    plt.title(f"Error Decay Curve over Horizons ({country})", fontsize=16)
    plt.xlabel("Forecast Horizon (Days)", fontsize=12)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
    plt.xticks(horizons)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(ARTIFACTS_DIR, f"error_decay_{country}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    print("Generating Evaluation Plots...")
    # Plot overlay for US and IN
    plot_overlay(country="US", horizon=1)
    plot_overlay(country="IN", horizon=1)
    
    # Plot error decay for US and IN
    plot_error_decay(country="US")
    plot_error_decay(country="IN")

if __name__ == "__main__":
    main()
