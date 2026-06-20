> ⚠️ **PROPRIETARY & CONFIDENTIAL**
> This repository contains the architectural implementation of the Global AQ Intelligence pipeline.
> The core V7 thermodynamic weights (`.pkl`), proprietary datasets, and historical telemetry databases are excluded to protect intellectual property.

# Global AQ Intelligence — ML Pipeline

[![Live Deployment](https://img.shields.io/badge/Live_Deployment-global--aq--intelligence.vercel.app-10B981?style=for-the-badge&logo=vercel)](https://global-aq-intelligence.vercel.app)
> **Currently running the V8 Horizon-Aligned Thermodynamics Engine.**

[📜 Read the full V8 Changelog & Architecture History here](CHANGELOG.md)

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

### Model Architecture: V8 Horizon-Aligned Thermodynamics Engine

**V8 Horizon-Aligned Architecture:**
We use strict autoregressive lags ($y_{t-h}$) and a 3-day rolling volatility matrix ($\sigma_{3d}$) engineered dynamically in Pandas prior to XGBoost inference. This completely eliminates the error compounding of recursive models.
- Independent models per horizon (h=1, 7, 14, 30) use lag features specifically aligned to their target horizon to prevent time leakage.
- Short-term memory and volatility metrics act as a momentum engine for high-variance regions.
- Thermodynamic modifiers (precipitation washout, wind dispersion) are applied via Open-Meteo future forecasts.

---

## Performance (V8 Horizon-Aligned Engine)

![Forecast Horizons EDA](./plots/forecast_horizons.png)

All metrics on held-out future data — strict chronological split, no leakage. We have officially deprecated the $R^2$ score in favor of Mean Absolute Scaled Error (MASE) and Normalized Mean Absolute Error (NMAE) due to the mathematical low-variance illusion in clean-air countries.

| Country | Code | Mean Absolute Error (MAE) | Real-World Accuracy (NMAE) | Intelligence Benchmark (MASE) |
| :--- | :--- | :--- | :--- | :--- |
| **India** | `IN` | 9.26 µg/m³ | **75.0%** | **0.88** (< 1.0) |
| **United States** | `US` | 2.24 µg/m³ | **65.3%** | **0.91** (< 1.0) |
| **Australia** | `AU` | 1.88 µg/m³ | **68.7%** | **0.85** (< 1.0) |
| **United Kingdom** | `GB` | 2.41 µg/m³ | **63.0%** | **0.94** (< 1.0) |

> **💡 The MASE Benchmark**
> MASE measures the model's accuracy against a naive baseline (predicting tomorrow will be identical to today). A score `< 1.0` proves the ML model is successfully outperforming the naive assumption. Our V8 model achieves MASE < 1.0 across all nodes.

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

---

For the full engineering history — data leakage discoveries, NASA POWER migration, thermodynamic interpolation design — see [`ISSUES.md`](./ISSUES.md).

---

### License & Copyright
© 2026 Divyansh Ailani. All Rights Reserved.
This code is provided strictly for **portfolio viewing and evaluation purposes**. You may not copy, modify, distribute, or run this pipeline without explicit permission.
