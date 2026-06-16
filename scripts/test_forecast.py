"""
Multi-day PM2.5 forecast validation.

Uses the v4 model to make recursive predictions for 1, 7, 14, and 30 days,
then compares against actual values to measure accuracy degradation over
forecast horizon.
"""

import joblib
import psycopg2
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load model
model = joblib.load('models/gb_pm25_v4_memory.pkl')
features = list(model.feature_names_in_)
print(f"Model loaded: {len(features)} features")


# Load data
conn = psycopg2.connect(
    host='localhost', dbname='indiaaq',
    user='postgres', password='8765'
)

df = pd.read_sql("""
    SELECT d.date, d.station_id, d.value as pm25,
           s.name as station_name, s.city,
           d.month, d.day_of_week, d.is_weekend, d.day_of_year,
           d.lag_1, d.lag_2, d.lag_3, d.lag_7,
           d.roll_3_mean, d.roll_7_mean, d.roll_3_std,
           d.temperature, d.humidity, d.wind_speed,
           d.no2_value, d.co_value, d.o3_value, d.so2_value
    FROM daily_features d
    JOIN stations s ON d.station_id = s.id
    WHERE d.parameter = 'pm25'
      AND d.value IS NOT NULL
      AND d.lag_7 IS NOT NULL
      AND d.temperature IS NOT NULL
    ORDER BY d.station_id, d.date
""", conn)
conn.close()

df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
print(f"Data: {len(df):,} rows, {df['station_id'].nunique()} stations")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


def recursive_forecast(station_df, start_idx, horizon, model, features):
    """
    Make recursive multi-day predictions.

    For day 1: use actual lag features.
    For day 2+: use previous prediction as lag_1, shift other lags.
    Weather features use actual values (in production, use weather forecast API).
    """
    predictions = []
    actuals = []
    dates = []

    # History buffer for computing rolling stats
    history = list(station_df.iloc[max(0, start_idx-7):start_idx]['pm25'].values)

    for step in range(horizon):
        idx = start_idx + step
        if idx >= len(station_df):
            break

        row = station_df.iloc[idx].copy()
        actuals.append(row['pm25'])
        dates.append(row['date'])

        if step == 0:
            # Day 1: use actual lag features from the database
            X = row[features].values.reshape(1, -1)
        else:
            # Day 2+: replace lag features with our predictions
            row_dict = row[features].to_dict()

            # Shift lags using our predictions and actual history
            pred_history = history + predictions
            if len(pred_history) >= 1:
                row_dict['lag_1'] = pred_history[-1]
            if len(pred_history) >= 2:
                row_dict['lag_2'] = pred_history[-2]
            if len(pred_history) >= 3:
                row_dict['lag_3'] = pred_history[-3]
            if len(pred_history) >= 7:
                row_dict['lag_7'] = pred_history[-7]

            # Recompute rolling stats from prediction history
            recent = pred_history[-3:] if len(pred_history) >= 3 else pred_history
            row_dict['roll_3_mean'] = np.mean(recent)
            row_dict['roll_3_std'] = np.std(recent) if len(recent) > 1 else 0

            recent_7 = pred_history[-7:] if len(pred_history) >= 7 else pred_history
            row_dict['roll_7_mean'] = np.mean(recent_7)

            X = np.array([[row_dict[f] for f in features]])

        pred = model.predict(X)[0]
        pred = max(0, pred)  # PM2.5 can't be negative
        predictions.append(pred)

    return np.array(predictions), np.array(actuals), dates


# Pick top 5 stations with most data
station_counts = df.groupby(['station_id', 'station_name']).size().reset_index(name='days')
station_counts = station_counts.sort_values('days', ascending=False)
top_stations = station_counts.head(5)

print(f"\nTop 5 stations by data availability:")
for _, row in top_stations.iterrows():
    print(f"  {row['station_name'][:40]:<40} {row['days']} days")


# Run forecasts for each horizon
horizons = [1, 7, 14, 30]
all_results = {h: {'mae': [], 'r2': []} for h in horizons}

print(f"\nRunning forecasts...")
print("-" * 70)

for _, srow in top_stations.iterrows():
    sid = srow['station_id']
    sname = srow['station_name'][:35]
    station_df = df[df['station_id'] == sid].sort_values('date').reset_index(drop=True)

    # Start from 70% of the way through (so we have enough future data)
    start = int(len(station_df) * 0.7)

    print(f"\n  {sname}")
    for h in horizons:
        preds, acts, dates = recursive_forecast(station_df, start, h, model, features)
        if len(preds) < h:
            print(f"    {h:>2}d: not enough data")
            continue

        mae = mean_absolute_error(acts, preds)
        r2 = r2_score(acts, preds) if len(acts) > 1 else float('nan')
        all_results[h]['mae'].append(mae)
        all_results[h]['r2'].append(r2)
        print(f"    {h:>2}d: MAE={mae:.1f} ug/m3, R2={r2:.3f}")


# Summary table
print("\n" + "=" * 50)
print("FORECAST ACCURACY BY HORIZON")
print("=" * 50)
print(f"{'Horizon':<12} {'Avg MAE':>10} {'Avg R2':>10} {'Verdict':>15}")
print("-" * 50)
for h in horizons:
    if all_results[h]['mae']:
        avg_mae = np.mean(all_results[h]['mae'])
        avg_r2 = np.mean(all_results[h]['r2'])
        if avg_mae < 10:
            verdict = "excellent"
        elif avg_mae < 20:
            verdict = "good"
        elif avg_mae < 40:
            verdict = "usable"
        else:
            verdict = "poor"
        print(f"{h:>2}-day{'':<7} {avg_mae:>10.1f} {avg_r2:>10.3f} {verdict:>15}")


# Plot: forecast for the best station at 30-day horizon
best_sid = top_stations.iloc[0]['station_id']
best_name = top_stations.iloc[0]['station_name'][:40]
best_df = df[df['station_id'] == best_sid].sort_values('date').reset_index(drop=True)
start = int(len(best_df) * 0.7)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, h in enumerate(horizons):
    ax = axes[i // 2, i % 2]
    preds, acts, dates = recursive_forecast(best_df, start, h, model, features)

    ax.plot(range(len(acts)), acts, 'b-', label='Actual', linewidth=1.5)
    ax.plot(range(len(preds)), preds, 'r--', label='Predicted', linewidth=1.5)

    mae = mean_absolute_error(acts, preds)
    ax.set_title(f'{h}-Day Forecast (MAE={mae:.1f})')
    ax.set_xlabel('Days')
    ax.set_ylabel('PM2.5 (ug/m3)')
    ax.legend()

plt.suptitle(f'PM2.5 Recursive Forecast: {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/forecast_horizons.png', dpi=150)
print(f"\nPlot saved: plots/forecast_horizons.png")
