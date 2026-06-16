"""
PM2.5 prediction model comparison.

Trains ML models on India's daily_features table, comparing
performance with and without lag features (temporal memory).
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load data
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

df = df.dropna()
print(f"After dropping NaN: {len(df):,} rows, {df['station_id'].nunique()} stations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nPM2.5 stats:")
print(df['pm25'].describe())


# Feature sets
features_no_memory = [
    'month', 'day_of_week', 'is_weekend', 'day_of_year',
    'temperature', 'humidity', 'wind_speed',
    'no2_value', 'co_value', 'o3_value', 'so2_value'
]

features_with_memory = features_no_memory + [
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'roll_3_mean', 'roll_7_mean', 'roll_3_std'
]

target = 'pm25'


# Chronological train/test split (no data leakage)
df = df.sort_values('date')
split_date = df['date'].quantile(0.8)

train = df[df['date'] <= split_date]
test = df[df['date'] > split_date]
print(f"\nTrain: {len(train):,} rows ({train['date'].min()} to {train['date'].max()})")
print(f"Test:  {len(test):,} rows ({test['date'].min()} to {test['date'].max()})")

y_train = train[target].values
y_test = test[target].values


def train_and_evaluate(feature_list, label):
    """Train all models on a given feature set and return results."""
    X_tr = train[feature_list].values
    X_te = test[feature_list].values

    results = {}
    print(f"\n{label}")
    print("-" * 60)

    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=200, max_depth=5, random_state=42)),
    ]:
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'pred': y_pred}
        print(f"  {name:<25} R2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")

    return results


# Experiment 1: without lag features
results_no_mem = train_and_evaluate(features_no_memory, "WITHOUT lag features (no memory)")

# Experiment 2: with lag features
results_with_mem = train_and_evaluate(features_with_memory, "WITH lag features (temporal memory)")


# Comparison
print("\nComparison")
print("-" * 65)
print(f"{'Model':<25} {'No Memory R2':>13} {'With Memory R2':>15} {'Delta':>8}")
print("-" * 65)
for name in results_no_mem:
    no_r2 = results_no_mem[name]['r2']
    mem_r2 = results_with_mem[name]['r2']
    print(f"{name:<25} {no_r2:>13.4f} {mem_r2:>15.4f} {mem_r2 - no_r2:>+8.4f}")


# Feature importance (best model)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gb.fit(train[features_with_memory].values, y_train)

importances = pd.Series(gb.feature_importances_, index=features_with_memory)
importances = importances.sort_values(ascending=False)

print("\nFeature importances (Gradient Boosting):")
for feat, imp in importances.head(10).items():
    print(f"  {feat:<18} {imp:.4f}")


# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs predicted
best_pred = results_with_mem['Gradient Boosting']['pred']
axes[0, 0].scatter(y_test, best_pred, alpha=0.1, s=5)
axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual PM2.5')
axes[0, 0].set_ylabel('Predicted PM2.5')
r2_gb = results_with_mem['Gradient Boosting']['r2']
axes[0, 0].set_title(f'Actual vs Predicted (R2={r2_gb:.3f})')

# R2 comparison
names = list(results_no_mem.keys())
no_mem_r2s = [results_no_mem[n]['r2'] for n in names]
mem_r2s = [results_with_mem[n]['r2'] for n in names]
x = np.arange(len(names))
axes[0, 1].bar(x - 0.15, no_mem_r2s, 0.3, label='No Memory', color='#ff6b6b')
axes[0, 1].bar(x + 0.15, mem_r2s, 0.3, label='With Memory', color='#51cf66')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
axes[0, 1].set_ylabel('R2 Score')
axes[0, 1].set_title('Effect of Temporal Features')
axes[0, 1].legend()

# Feature importance
importances.head(10).plot(kind='barh', ax=axes[1, 0], color='#339af0')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Features')
axes[1, 0].invert_yaxis()

# Time series: actual vs predicted for one station
sample_station = test['station_id'].value_counts().index[0]
sample = test[test['station_id'] == sample_station].head(60)
sample_pred = gb.predict(sample[features_with_memory].values)
axes[1, 1].plot(range(len(sample)), sample['pm25'].values, 'b-', label='Actual', linewidth=1.5)
axes[1, 1].plot(range(len(sample)), sample_pred, 'r--', label='Predicted', linewidth=1.5)
axes[1, 1].set_xlabel('Days')
axes[1, 1].set_ylabel('PM2.5 (ug/m3)')
axes[1, 1].set_title(f'60-Day Prediction: Station {sample_station}')
axes[1, 1].legend()

plt.suptitle('PM2.5 Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/model_comparison_memory.png', dpi=150)
print(f"\nPlot saved: plots/model_comparison_memory.png")
