> ⚠️ **PROPRIETARY & CONFIDENTIAL**
> This repository contains the architectural implementation of the Global AQ Intelligence pipeline.
> The core V7 thermodynamic weights (`.pkl`), proprietary datasets, and historical telemetry databases are excluded to protect intellectual property.

# Global AQ Intelligence — ML Pipeline

[![Live Deployment](https://img.shields.io/badge/Live_Deployment-global--aq--intelligence.vercel.app-10B981?style=for-the-badge&logo=vercel)](https://global-aq-intelligence.vercel.app)
> **Currently running the V11 3D Atmospheric Ensemble Router.**

[📜 Read the full V11 Changelog & Architecture History here](CHANGELOG.md)

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

### Model Architecture: V11 3D Atmospheric Ensemble Router

**V11 Global Unified Architecture (Native XGBoost):**
We migrated from a single model to a dynamic ensemble router. The core mathematical foundation builds on the V11 XGBoost engine with several major enhancements:
- **Dynamic Routing**: Great Britain relies on the V9 physics-backed persistence model for long-term horizons, while all other regions/horizons use the V11 engine.
- **Delta Target Transformation ($\Delta Y$)**: The engine predicts 'Velocity' ($\Delta Y = Y_t - Y_{t-1}$) to force the model to explicitly correct the naive baseline, unlocking significant long-term stability.
- **SUOMI VIIRS Spatial 'Blast Radius' Engine**: Uses the Haversine formula to bridge satellite fire coordinates with ground stations, creating a 100km `fire_density` and `fire_radiative_power` dynamic feature set.
- **Fading Memory (EMA)**: An Exponential Moving Average (EMA) gives higher weight to recent micro-fluctuations, crushing the 1-day horizon underfitting problem.
- **3D Atmospheric Depth (AOD)**: Live injection of Aerosol Optical Depth from Open-Meteo satellite arrays to physically map vertical pollution density.

---

## Performance (V11 Geospatial Ensemble)

### V11 → V11.1: The MASE Crusher Upgrade

The V11.1 engine represents a fundamental shift from "Single-Day Myopia" to **Autonomous Atmospheric Physics**. Instead of only looking at today's weather, the model now understands *cumulative* weather patterns over 72 hours and atmospheric instability over a trailing week.

#### V11 (Old — Before Physics Memory)
![V11 Old Performance](./plots/forecast_horizons_v11_old.png)

#### V11.1 (New — Autonomous Atmospheric Physics)
![V11.1 New Performance](./plots/forecast_horizons_v11_new.png)

#### What Changed (The Physics Upgrade)

| Feature | V11 (Old) | V11.1 (New) |
| :--- | :--- | :--- |
| Rain Awareness | Single-day `future_precip` only | `rolling_3day_precip` — 72hr cumulative rain memory |
| Atmospheric Stability | None | `aod_volatility_index` — 7-day AOD standard deviation |
| Heuristic Overrides | `is_raining_now` hardcoded rules | ❌ Completely purged — model learns autonomously |
| Optimization Target | MASE (basic) | MASE (20-trial Optuna per country × horizon) |
| Matrix Rebuild | N/A | 1,627,674 rows via PostgreSQL Window Functions |

**The Key Insight:** The V11 model had zero "Atmospheric Memory." It didn't know that 72 hours of continuous monsoon rain had already flushed all particulate matter from the atmosphere. During the June 25th India anomaly, V11 predicted 49.44 µg/m³ while the ground truth was 14.61 µg/m³. After the V11.1 physics upgrade, the model **natively** learned to drop its prediction by ~20 µg/m³ during heavy rain events (58.75 → 38.74) without any hardcoded rules.

> **💡 The Mean Reversion Trap (Issue #18)**
> The remaining gap from 38.74 to 14.61 µg/m³ is a fundamental property of MAE-optimized decision trees — they hedge toward the average of all rain-day outcomes rather than predicting extreme tail events. Two future ML architectures are identified: **Quantile Regression** and **Two-Tier Regime-Switching** (see [`ISSUES.md`](./ISSUES.md#18)).

#### V11.1 Optuna-Optimized Metrics (Global)

| Country | Horizon | MAE | NMAE | MASE | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IN** | 1 | 9.76 | 0.2542 | 0.8900 | **74.58%** |
| **IN** | 7 | 15.72 | 0.4046 | 0.6800 | **59.54%** |
| **IN** | 14 | 16.89 | 0.4364 | 0.6000 | **56.36%** |
| **IN** | 30 | 15.60 | 0.4106 | 0.5100 | **58.94%** |
| **GB** | 1 | 2.15 | 0.3369 | 0.8200 | **66.31%** |
| **GB** | 7 | 2.41 | 0.3798 | 0.5700 | **62.02%** |
| **GB** | 14 | — | — | — | V9 Fallback |
| **GB** | 30 | — | — | — | V9 Fallback |
| **US** | 1 | 2.21 | 0.3025 | 0.8300 | **69.75%** |
| **US** | 7 | 2.88 | 0.3934 | 0.7100 | **60.66%** |
| **US** | 14 | 2.98 | 0.4072 | 0.7000 | **59.28%** |
| **US** | 30 | 3.02 | 0.4125 | 0.7100 | **58.75%** |
| **AU** | 1 | 1.83 | 0.3025 | 0.7800 | **69.75%** |
| **AU** | 7 | 2.14 | 0.3540 | 0.6600 | **64.60%** |
| **AU** | 14 | 2.10 | 0.3465 | 0.6500 | **65.35%** |
| **AU** | 30 | 2.16 | 0.3571 | 0.6600 | **64.29%** |

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
│   ├── build_global_features.py  # Bulk feature backfill
│   └── backfill_aod_partitioned.py # Multi-VM parallel AOD backfill
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
| v9.4 | Geospatial Ensemble | Delta Target Transformation, VIIRS Spatial Blast Radius, EMA Fading Memory |
| v11 | 3D Atmospheric Ensemble | 3D Aerosol Optical Depth (AOD) via Open-Meteo Satellite Sync |
| v11.1 | Autonomous Physics Engine | Atmospheric Memory (`rolling_3day_precip`), AOD Volatility, Optuna MASE Crusher, heuristic purge |

---

## 🚧 Ongoing Issues & Data Health

The following issues were identified during a full Azure DB audit (2026-06-27) and are actively being tracked on GitHub:

| # | Issue | Status | Impact |
|---|-------|--------|--------|
| [#5](https://github.com/divyanshailani/global-aq-intelligence-pipeline/issues/5) | Environment State Divergence — Local vs Azure DB | ✅ Resolved | Data sync fixed |
| [#1](https://github.com/divyanshailani/global-aq-intelligence-pipeline/issues/1) | 13 Legacy Columns at 95% NULL | ✅ Resolved | 11 columns dropped via CASCADE |
| [#2](https://github.com/divyanshailani/global-aq-intelligence-pipeline/issues/2) | 1,464 Phantom Stations (35%) with zero features | 🔍 Investigation | ETL coverage gap |
| [#4](https://github.com/divyanshailani/global-aq-intelligence-pipeline/issues/4) | Empty `model_registry` & `predictions` tables | 🔍 Investigation | Schema cleanup |
| [#3](https://github.com/divyanshailani/global-aq-intelligence-pipeline/issues/3) | AOD Backfill: `om_aerosol_optical_depth` at 33% NULL | 🔧 In Progress | 4-node parallel backfill running |

### 🖥️ Multi-VM Backfill Architecture

Currently running a 4-node parallel AOD backfill to fill ~540K NULL satellite records:

```
┌─────────────────────────────────────────────────────────────┐
│                    Open-Meteo Air Quality API               │
│              (10K requests/day per IP limit)                 │
└──────┬──────────┬──────────┬──────────┬──────────────────────┘
       │          │          │          │
  IP₁  │    IP₂   │    IP₃   │    IP₄   │
       ▼          ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│  DO VM 1 ││Azure B1s ││  DO VM 2 ││ Mac Mini │
│ Part 0/4 ││ Part 1/4 ││ Part 2/4 ││ Part 3/4 │
│ ~453 stn ││ ~453 stn ││ ~453 stn ││ ~453 stn │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
                      │
                      ▼
     ┌──────────────────────────────┐
     │   Azure PostgreSQL Flex      │
     │   (50 max connections)       │
     │   UPDATE ... WHERE IS NULL   │
     └──────────────────────────────┘
```

Each VM runs `backfill_aod_partitioned.py --partition N --total 4` in a tmux session.

> For the full database audit report, see [`CHANGELOG.md`](CHANGELOG.md) entry `[11.1.3]`.

For the full engineering history — data leakage discoveries, NASA POWER migration, thermodynamic interpolation design — see [`ISSUES.md`](./ISSUES.md).

---

### License & Copyright
© 2026 Divyansh Ailani. All Rights Reserved.
This code is provided strictly for **portfolio viewing and evaluation purposes**. You may not copy, modify, distribute, or run this pipeline without explicit permission.
