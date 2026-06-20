> ⚠️ **PROPRIETARY & CONFIDENTIAL**
> This repository contains the architectural implementation of the Global AQ Intelligence pipeline.
> The core V7 thermodynamic weights (`.pkl`), proprietary datasets, and historical telemetry databases are excluded to protect intellectual property.

# Global AQ Intelligence — ML Pipeline

[![Live Deployment](https://img.shields.io/badge/Live_Deployment-global--aq--intelligence.vercel.app-10B981?style=for-the-badge&logo=vercel)](https://global-aq-intelligence.vercel.app)
> **Currently running the V9 XGBoost Thermodynamics Engine.**

[📜 Read the full V9 Changelog & Architecture History here](CHANGELOG.md)

![Dashboard Screenshot](https://raw.githubusercontent.com/divyanshailani/global-aq-intelligence-web/main/public/images/ui_dashboard.png)
> End-to-end PM2.5 forecasting engine for 4 countries. Autonomous daily pipeline: fetch → engineer → predict → export → sync.

**Stack:** Python · PostgreSQL · scikit-learn GBR · NASA POWER · Open-Meteo · FastAPI

**Frontend:** [global-aq-intelligence-web](https://github.com/divyanshailani/global-aq-intelligence-web)

---



## What It Does

Predicts PM2.5 air pollution for India, USA, UK, and Australia at 1-day, 7-day, 14-day, and 30-day horizons using a Gradient Boosting Regressor with a physics-based weather interpolation layer.

One command runs the full pipeline end-to-end:

```bash
python3 scripts/predict_pipeline.py
```

This fetches live sensor data, generates 30-day forecasts per station, exports static JSON, and automatically syncs to the Next.js frontend.

---

## Architecture

```
OpenAQ API ──┐
NASA POWER ──┼──▶ PostgreSQL ──▶ Feature Engineering ──▶ V7 Models ──▶ JSON Export ──▶ Next.js
Open-Meteo ──┘                   (lag/rolling/delta)     (GBR × 16)    (site_data/)    (auto-sync)
```

### Model Architecture: V9 XGBoost Thermodynamics Engine

**V9 Global Unified Architecture (Native XGBoost):**
We migrated from scikit-learn GBR to native XGBoost (DMatrix) to achieve a 46x compute speedup by bypassing Python loops in favor of hardware-level C++ matrix operations. The core mathematical foundation remains unchanged:
- **Horizon-Aligned Lags ($y_{t-h}$)**: Each of the 16 independent models aligns its autoregressive lags strictly to its forecast horizon to eliminate time leakage.
- **3-Day Rolling Volatility Matrix ($\sigma_{3d}$)**: Acts as a momentum engine for high-variance regions.
- **Thermodynamic Modifiers**: Applied via Open-Meteo future forecasts (precipitation washout, wind dispersion).

---

## Performance (V9 XGBoost Engine)

![Forecast Horizons EDA](./plots/forecast_horizons.png?v=9)

The XGBoost models yielded phenomenal speed-ups and maintained robust efficiency boundaries across all anchor horizons globally. Highlight: maintaining ~56-62% accuracy even at 30-day horizons in chaotic environments.

| Country | Horizon | MAE | NMAE | MASE | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IN** | 1 | 10.41 | 0.2748 | 0.9600 | **72.52%** |
| **IN** | 7 | 16.33 | 0.4245 | 0.7100 | **57.55%** |
| **IN** | 14 | 17.19 | 0.4462 | 0.6100 | **55.38%** |
| **IN** | 30 | 15.79 | 0.4175 | 0.5200 | **58.25%** |
| **GB** | 1 | 2.26 | 0.3563 | 0.8500 | **64.37%** |
| **GB** | 7 | 2.59 | 0.4063 | 0.6400 | **59.37%** |
| **GB** | 14 | 2.78 | 0.4401 | 0.5700 | **55.99%** |
| **GB** | 30 | 2.65 | 0.4333 | 0.6400 | **56.67%** |
| **US** | 1 | 2.35 | 0.3215 | 0.8800 | **67.85%** |
| **US** | 7 | 3.13 | 0.4278 | 0.7700 | **57.22%** |
| **US** | 14 | 3.22 | 0.4401 | 0.7600 | **55.99%** |
| **US** | 30 | 3.18 | 0.4352 | 0.7500 | **56.48%** |
| **AU** | 1 | 1.91 | 0.3158 | 0.8100 | **68.42%** |
| **AU** | 7 | 2.19 | 0.3626 | 0.6700 | **63.74%** |
| **AU** | 14 | 2.16 | 0.3572 | 0.6700 | **64.28%** |
| **AU** | 30 | 2.24 | 0.3701 | 0.6800 | **62.99%** |

---

## Project Structure

```
.
├── scripts/
│   ├── predict_pipeline.py        # Main: fetch → predict → export → sync
│   ├── train_v5.py                # Legacy chained GBR (baseline)
│   ├── train_v6.py                # Direct multi-horizon (no future weather)
│   ├── train_v7_experiment.py     # V7: direct + future weather injection
│   ├── fetch_openaq.py            # Live sensor data
│   ├── fetch_nasa_power.py        # Historical satellite weather
│   ├── fetch_firms_fire.py        # NASA FIRMS fire count data
│   ├── cleanup_prediction_log.py  # Archive impossible past-date rows
│   └── build_global_features.py  # Bulk feature backfill
├── src/
│   ├── config.py                  # DB config + paths
│   ├── features.py                # Feature engineering (lag/rolling/delta)
│   ├── cleaning.py                # Outlier removal + null handling
│   └── aggregations.py            # Station-level daily aggregation
├── models/
│   ├── v5/                        # Legacy (chained) — kept as baseline
│   ├── v6/                        # Direct horizon — no future weather
│   └── v7/                        # Production — direct + future weather
├── sql/
│   └── schema.sql                 # Schema + v6 migration (ADD COLUMN IF NOT EXISTS)
├── data/
│   └── site_data/                 # Exported JSONs (auto-synced to frontend)
├── tests/
│   ├── test_codex_fixes.py
│   └── test_processing.py
├── ISSUES.md                      # Engineering log — 8 problems and how they were solved
├── requirements.txt
└── .env.example
```

---

## Feature Engineering

All features are strictly backward-looking. No same-day or future values in training.

| Group | Features | Rationale |
|-------|----------|----------|
| Short lags | lag_1, lag_2, lag_3, lag_7 | Recent pollution memory |
| Long lags | lag_14, lag_21, lag_30 | Monthly context, seasonal baseline |
| Rolling | roll_3/7/14/30_mean, roll_3/14_std | Trend + volatility |
| Momentum | pm25_delta_1, pm25_delta_7 | Rising vs falling signal |
| Weather (hist) | temperature, humidity, wind_speed (NASA POWER) | Dispersion conditions |
| Weather (future) | future_temp, future_wind, future_precip (Open-Meteo) | V7 thermodynamics |
| Pollutants | no2, co, o3, so2 (lagged) | Chemical co-occurrence |
| Fire | fire_count (NASA FIRMS) | Wildfire contribution |
| Calendar | month, day_of_week, day_of_year, is_weekend | Seasonal + traffic cycles |

---

## Running Locally

**Prerequisites:** PostgreSQL 15+, Python 3.11+

```bash
# 1. Clone and install
git clone https://github.com/divyanshailani/global-aq-intelligence-pipeline
cd global-aq-intelligence-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set up database
createdb indiaaq
psql indiaaq < sql/schema.sql

# 3. Configure environment
cp .env.example .env
# Fill in DB credentials

# 4. Run the full pipeline
python3 scripts/predict_pipeline.py

# 5. Skip fetch (use existing DB data)
python3 scripts/predict_pipeline.py --skip-fetch

# 6. Retrain V7 models
python3 scripts/train_v7_experiment.py
```

Output JSONs are written to `data/site_data/` and automatically synced to `../global-aq-intelligence/public/data/` if the frontend repo is present on the same machine.

---

## Model Version History

| Version | Strategy | Key Change |
|---------|----------|------------|
| v5 | Chained GBR | 30-day loop feeding predictions as lag inputs |
| v6 | Direct multi-horizon | Separate model per horizon, no chaining |
| v7 | Direct + future weather | Open-Meteo 16-day forecast injected at inference |
| v8 | Global Unified | Horizon-Aligned Lags & Volatility Matrix |
| v9 | Global Unified | Native XGBoost, Horizon-Aligned Lags & Volatility Matrix |

---

For the full engineering history — data leakage discoveries, NASA POWER migration, thermodynamic interpolation design — see [`ISSUES.md`](./ISSUES.md).

---

### License & Copyright
© 2026 Divyansh Ailani. All Rights Reserved.
This code is provided strictly for **portfolio viewing and evaluation purposes**. You may not copy, modify, distribute, or run this pipeline without explicit permission.
