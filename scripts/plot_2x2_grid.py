import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PREDICTIONS_DIR = os.path.join("data", "predictions_v12")
ARTIFACTS_DIR = "/Users/divyanshailani/.gemini/antigravity/brain/f18c2b27-d05c-474b-8f60-58bdc6cc3c31"

def plot_2x2_v12_pure(country, station_id):
    model_name = "V12 (Challenger Pure)"
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    horizons = [1, 7, 14, 30]
    
    fig.suptitle(f"{model_name} Forecast: Station {station_id} ({country})", fontsize=18, fontweight='bold', y=0.95)
    
    for i, h in enumerate(horizons):
        file_path = os.path.join(PREDICTIONS_DIR, f"eval_v12_{country}_{h}.csv")
        ax = axes[i]
        
        if not os.path.exists(file_path):
            ax.text(0.5, 0.5, f"No Data for {h}d", ha='center', va='center', fontsize=14)
            ax.set_title(f"{h}-Day Forecast")
            continue
            
        df = pd.read_csv(file_path)
        # Filter and sort by the FUTURE target date for proper alignment
        df_station = df[df['station_id'] == station_id].sort_values('target_date').reset_index(drop=True)
        
        if len(df_station) == 0:
            ax.text(0.5, 0.5, f"No Station Data", ha='center', va='center', fontsize=14)
            ax.set_title(f"{h}-Day Forecast")
            continue
            
        y_act = df_station['actual_future_pm25']
        y_pred = df_station['v12_pred']
        
        mask = y_act.notna() & y_pred.notna()
        y_act = y_act[mask]
        y_pred = y_pred[mask]
        
        if len(y_act) == 0:
            ax.text(0.5, 0.5, f"No Valid Predictions", ha='center', va='center', fontsize=14)
            continue
            
        mae = np.mean(np.abs(y_act - y_pred))
        
        x_vals = np.arange(len(y_act))
        ax.plot(x_vals, y_act, label='Actual (Future)', color='blue', linewidth=1.5)
        ax.plot(x_vals, y_pred, label='Predicted', color='green', linestyle='--', linewidth=1.5)
        
        ax.set_title(f"{h}-Day Forecast (MAE={mae:.1f})", fontsize=12)
        ax.set_xlabel("Days (Holdout Split Chronological)", fontsize=10)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=10)
        ax.legend(fontsize=10)
        
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(ARTIFACTS_DIR, f"grid_pure_v12_{country}_{station_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    plot_2x2_v12_pure("IN", 693)
    plot_2x2_v12_pure("US", 2775)

if __name__ == "__main__":
    main()
