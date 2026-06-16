"""
Global AQ Pipeline — PM2.5 Prediction with Lag Features (Fake Memory)
=====================================================================
This script trains ML models on India's daily_features table,
comparing performance WITH and WITHOUT lag features to prove
that "fake memory" improves predictions.
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data from PostgreSQL ──────────────────────────
print("Loading data from PostgreSQL...")
conn = psycopg2.connect(
    host='localhost', dbname='indiaaq',
    user='postgres', password='8765'
)

df = pd.read_sql("""
    SELECT date, station_id, value as pm25,
           month, day_of_week, is_weekend, day_of_year,
           lag_1, lag_2, lag_3, lag_7,
           roll_3_mean, roll_7_mean, roll_3_std,
           temperature, humidity, wind_speed,
           no2_value, co_value, o3_value, so2_value
    FROM daily_features
    WHERE parameter = 'pm25'
      AND value IS NOT NULL
      AND lag_7 IS NOT NULL
      AND temperature IS NOT NULL
    ORDER BY station_id, date
""", conn)
conn.close()

print(f"Loaded: {len(df):,} rows, {df['station_id'].nunique()} stations")

# Drop rows with any NaN in features
df = df.dropna()
print(f"After dropping NaN: {len(df):,} rows, {df['station_id'].nunique()} stations")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"\nTarget (PM2.5) stats:")
print(df['pm25'].describe())

# ── 2. Define Feature Sets ────────────────────────────────
# WITHOUT lag features (model has NO memory)
features_no_memory = [
    'month', 'day_of_week', 'is_weekend', 'day_of_year',
    'temperature', 'humidity', 'wind_speed',
    'no2_value', 'co_value', 'o3_value', 'so2_value'
]

# WITH lag features (model has FAKE memory)
features_with_memory = features_no_memory + [
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'roll_3_mean', 'roll_7_mean', 'roll_3_std'
]

target = 'pm25'

# ── 3. Time-based split (IMPORTANT for time series!) ──────
# DON'T use random split for time data — use chronological split
# Otherwise future data leaks into training
df = df.sort_values('date')
split_date = df['date'].quantile(0.8)

train = df[df['date'] <= split_date]
test = df[df['date'] > split_date]
print(f"\nTrain: {len(train):,} rows ({train['date'].min()} → {train['date'].max()})")
print(f"Test:  {len(test):,} rows ({test['date'].min()} → {test['date'].max()})")

# ── 4. Train Models: WITHOUT Memory ──────────────────────
print("\n" + "="*60)
print("EXPERIMENT 1: WITHOUT LAG FEATURES (no memory)")
print("="*60)

X_train_no = train[features_no_memory].values
X_test_no = test[features_no_memory].values
y_train = train[target].values
y_test = test[target].values

models_no_memory = {}
for name, model in [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)),
]:
    model.fit(X_train_no, y_train)
    y_pred = model.predict(X_test_no)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    models_no_memory[name] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'pred': y_pred}
    print(f"  {name:<25} R²={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")

# ── 5. Train Models: WITH Memory ─────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 2: WITH LAG FEATURES (fake memory)")
print("="*60)

X_train_mem = train[features_with_memory].values
X_test_mem = test[features_with_memory].values

models_with_memory = {}
for name, model in [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)),
]:
    model.fit(X_train_mem, y_train)
    y_pred = model.predict(X_test_mem)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    models_with_memory[name] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'pred': y_pred}
    print(f"  {name:<25} R²={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")

# ── 6. Comparison ────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON: Memory Effect")
print("="*60)
print(f"{'Model':<25} {'No Memory R²':>13} {'With Memory R²':>15} {'Improvement':>12}")
print("-" * 65)
for name in models_no_memory:
    no_r2 = models_no_memory[name]['r2']
    mem_r2 = models_with_memory[name]['r2']
    improvement = mem_r2 - no_r2
    print(f"{name:<25} {no_r2:>13.4f} {mem_r2:>15.4f} {improvement:>+12.4f}")

# ── 7. Feature Importance (best model) ───────────────────
best_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
best_model.fit(X_train_mem, y_train)

importances = pd.Series(best_model.feature_importances_, index=features_with_memory)
importances = importances.sort_values(ascending=False)

print(f"\nTop 10 Feature Importances (Gradient Boosting):")
for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<18} {imp:.4f}  {bar}")

# ── 8. Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted (with memory, best model)
best_pred = models_with_memory['Gradient Boosting']['pred']
axes[0, 0].scatter(y_test, best_pred, alpha=0.1, s=5)
axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual PM2.5')
axes[0, 0].set_ylabel('Predicted PM2.5')
axes[0, 0].set_title(f'Actual vs Predicted (R²={models_with_memory["Gradient Boosting"]["r2"]:.3f})')

# Plot 2: R² comparison bar chart
names = list(models_no_memory.keys())
no_mem_r2s = [models_no_memory[n]['r2'] for n in names]
mem_r2s = [models_with_memory[n]['r2'] for n in names]
x = np.arange(len(names))
axes[0, 1].bar(x - 0.15, no_mem_r2s, 0.3, label='No Memory', color='#ff6b6b')
axes[0, 1].bar(x + 0.15, mem_r2s, 0.3, label='With Memory', color='#51cf66')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('Memory Effect: R² Comparison')
axes[0, 1].legend()

# Plot 3: Feature importance
importances.head(10).plot(kind='barh', ax=axes[1, 0], color='#339af0')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Features (Gradient Boosting)')
axes[1, 0].invert_yaxis()

# Plot 4: Time series sample — actual vs predicted for 1 station
sample_station = test['station_id'].value_counts().index[0]
sample = test[test['station_id'] == sample_station].head(60)
sample_idx = sample.index
sample_pred = best_model.predict(sample[features_with_memory].values)
axes[1, 1].plot(range(len(sample)), sample['pm25'].values, 'b-', label='Actual', linewidth=1.5)
axes[1, 1].plot(range(len(sample)), sample_pred, 'r--', label='Predicted', linewidth=1.5)
axes[1, 1].set_xlabel('Days')
axes[1, 1].set_ylabel('PM2.5 (µg/m³)')
axes[1, 1].set_title(f'60-Day Forecast: Station {sample_station}')
axes[1, 1].legend()

plt.suptitle('Global AQ Pipeline — India PM2.5 Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/model_comparison_memory.png', dpi=150)
print(f"\nPlot saved: plots/model_comparison_memory.png")
