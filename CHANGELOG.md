# Changelog

All notable changes to this project will be documented in this file.

## [11.0.0] - V11 3D Atmospheric Ensemble Active

### 🚀 Recent Updates (V11)
- **3D Aerosol Optical Depth (AOD) Fusion**: Interfaced with Open-Meteo's CAMS European satellite data to extract a 3D physical representation of atmospheric aerosols.
- **Extreme Spike Elimination**: Achieved a breakthrough MAE of 76.06 µg/m³ on extreme spikes (True PM2.5 > 150), compared to the V9.4 baseline of 87.48.
- **Global Deployment**: Upgraded the dynamic router to dispatch all short-horizon inferences globally through the V11 physics engine. Great Britain long horizons safely fallback to V9.
- **Per-Country Optuna Tuning**: Executed an independent Optuna hyperparameter sweep for each geographic zone. This dynamic matrix allows the architecture to constrain itself with shallow trees (`max_depth=6`) in low-variance zones like Australia, while utilizing deeper trees (`max_depth=9`) to map complex 3D AOD patterns in high-variance zones like India.

## [9.4.0] - V9.4 Geospatial Ensemble Router

### 🚀 Recent Updates (V9.4)
- **Dynamic Geospatial Ensemble Router**: Implemented a sophisticated routing architecture that dynamically dispatches prediction requests. Great Britain uses V9 for long horizons ($h=14, 30$), while all other regions and short horizons use the V9.4 engine.
- **Delta Target Transformation**: Pivoted the target prediction to 'Velocity' ($\Delta Y = Y_t - Y_{t-1}$) to force the model to explicitly correct the naive baseline, unlocking significant long-term stability for US and AU nodes.
- **SUOMI VIIRS Spatial 'Blast Radius' Engine**: Bridged the spatial gap between satellite fire coordinates and ground-based AQI stations. The engine queries NASA FIRMS (VIIRS) to compute a 100km `fire_density` and `fire_radiative_power` dynamically.
- **Fading Memory (EMA)**: Upgraded from a simple 3-day rolling mean to an Exponential Moving Average (EMA) to give higher weight to recent micro-fluctuations, crushing the 1-day horizon underfitting problem (India $h=1$ MASE dropped from 0.96 to 0.90).


## [8.0.0] - V8 Direct-Horizon Engine (Deprecated)

### 🚀 Recent Updates (Transition to V8)
- **Horizon-Aligned Autoregressive Memory**: Shifted from a stateless model to a time-aware engine by injecting strict $y_{t-h}$ lags and $\sigma_{3d}$ volatility matrices, entirely eliminating the high-variance MASE trap in chaotic environments like India and GB.
- **Metric Engineering (NMAE & MASE)**: Deprecated the mathematically flawed $R^2$ score for low-variance environments. The backend now strictly evaluates using MASE (Mean Absolute Scaled Error) against a naive baseline, while the UI renders a normalized Accuracy Percentage (NMAE).
- **ETL Fault Tolerance**: Engineered a resilient asynchronous batching pipeline (`aiohttp`). The system successfully demonstrated zero-downtime fault tolerance by seamlessly falling back to internal database states when the primary OpenAQ API threw a 401 Unauthorized suspension block.
- **UX Time-Relativity**: Eradicated the 'Index 0' date trap on the Next.js frontend by implementing strict mathematical matching between the user's local browser `new Date()` and the payload's absolute timestamp.

### Historical Architecture: V8 Horizon-Aligned Thermodynamics Engine

**V8 Horizon-Aligned Architecture (scikit-learn GBR):**
We used strict autoregressive lags ($y_{t-h}$) and a 3-day rolling volatility matrix ($\sigma_{3d}$) engineered dynamically in Pandas prior to inference. This completely eliminated the error compounding of recursive models.
- Independent models per horizon (h=1, 7, 14, 30) used lag features specifically aligned to their target horizon to prevent time leakage.
- Short-term memory and volatility metrics acted as a momentum engine for high-variance regions.
- Thermodynamic modifiers (precipitation washout, wind dispersion) were applied via Open-Meteo future forecasts.

### Historical Performance (V8 Horizon-Aligned Engine)

All metrics on held-out future data — strict chronological split, no leakage. We officially deprecated the $R^2$ score in favor of Mean Absolute Scaled Error (MASE) and Normalized Mean Absolute Error (NMAE) due to the mathematical low-variance illusion in clean-air countries.

| Country | Code | Mean Absolute Error (MAE) | Real-World Accuracy (NMAE) | Intelligence Benchmark (MASE) |
| :--- | :--- | :--- | :--- | :--- |
| **India** | `IN` | 9.26 µg/m³ | **75.0%** | **0.88** (< 1.0) |
| **United States** | `US` | 2.24 µg/m³ | **65.3%** | **0.91** (< 1.0) |
| **Australia** | `AU` | 1.88 µg/m³ | **68.7%** | **0.85** (< 1.0) |
| **United Kingdom** | `GB` | 2.41 µg/m³ | **63.0%** | **0.94** (< 1.0) |

> **💡 The MASE Benchmark**
> MASE measures the model's accuracy against a naive baseline (predicting tomorrow will be identical to today). A score `< 1.0` proves the ML model is successfully outperforming the naive assumption. Our V8 model achieves MASE < 1.0 across all nodes.

---

## [7.0.0] - V7 Thermodynamics Engine

### Historical Architecture: V7 Direct Thermodynamics Engine

The core insight from v5 development: chaining Day-1 predictions into Day-2's lag features compounds error exponentially. By Day-30 the model was predicting noise.

**V7 fix — direct horizon models:**
- Train one independent GBR per horizon per country (4 countries × 4 horizons = 16 models)
- Each model predicts directly from real observed features — no chaining, no error propagation
- Anchor points: h1, h7, h14, h30 are direct model outputs
- Intermediate days (2–6, 8–13, 15–29): weather-weighted interpolation between anchors

**Thermodynamic modifiers applied during interpolation:**
- Rain washout: precipitation > 2mm → PM2.5 reduced 30%
- Wind dispersion: wind > 15 km/h → PM2.5 reduced 15%
- Stagnation spike: wind < 5 km/h + no precip → PM2.5 increased 20%

**V7 feature additions over V6:**
- `future_temp`, `future_wind`, `future_precip` — injected from Open-Meteo 16-day forecast per station per date
- Falls back to station climatology baseline for horizons beyond 15 days

### Historical Performance (V7 Thermodynamics Engine)

| Country | Code | R² Score | Mean Absolute Error (MAE) | Real-World Accuracy (NMAE) |
| :--- | :--- | :--- | :--- | :--- |
| **India** | `IN` | 0.750 | 9.26 µg/m³ | **66.1%** |
| **United States** | `US` | 0.499 | 0.84 µg/m³ | **84.1%** |
| **Australia** | `AU` | 0.451 | 1.57 µg/m³ | **70.5%** |
| **United Kingdom** | `GB` | 0.248 | 2.41 µg/m³ | **63.0%** |

> **💡 Architect's Note: The Low-Variance Trap & NMAE**
> You might notice a discrepancy between $R^2$ and MAE in developed nations (like the US and GB). Because their raw PM2.5 levels are extremely low and stable (low variance), the $R^2$ formula mathematically penalizes the model disproportionately for tiny micro-errors. 
> To counter this and provide a truthful confidence score for the UI, this engine calculates **Normalized Mean Absolute Error (NMAE)** against the historical mean, converting it into a robust real-world Accuracy %.

**Known weaknesses (tracked in `ISSUES.md`):**
- `value` (today's PM2.5) holds ~82% feature importance for h1 India — the model is a physics-backed persistence model at short range
- US h7 R²=0.14 — a single country-level GBR is too coarse for 1,400 geographically diverse stations
- India h30 overfit delta = 0.45 (train R²=0.88 vs test R²=0.43)

---

## [6.0.0] - V6 Direct Multi-Horizon
- **Direct Multi-Horizon Architecture**: Separated the model into 4 independent GBRs per country targeting specific horizons (`1d`, `7d`, `14d`, `30d`). 
- Completely abandoned the chained looping mechanism to prevent recursive error compounding.

---

## [5.0.0] - V5 Chained GBR
- **Recursive 30-Day Loop**: Implemented a recursive prediction loop feeding Day 1's prediction as `lag_1` into Day 2.
- **The Chaining Collapse**: Led to exponential error degradation; predictions beyond day 15 became pure noise.

---

## [4.0.0] - Temporal Memory Features
- **Temporal Memory Injection**: Engineered a robust suite of temporal features including short/long lags (`lag_1` to `lag_30`), rolling means, and rolling volatility.
- **Massive Performance Gain**: Short-term $R^2$ skyrocketed from ~0.31 to 0.97 by giving the model memory of yesterday's air quality.

---

## [3.0.0] - NASA POWER Migration
- **Open-Meteo Null Rate Failure**: Discarded Open-Meteo for historical training data due to a catastrophic 63% null rate in Indian coverage areas.
- **Satellite Data Integration**: Successfully integrated the NASA POWER API (satellite-derived global meteorology), plummeting the missing data rate to 0.15%.

---

## [2.0.0] - Co-Pollutant Integration
- **Chemical Context**: Added co-pollutants ($NO_2$, $CO$, $O_3$, $SO_2$) to the feature set.
- **Data Leakage Fix**: Prevented "cheating" by strictly shifting chemical context backward (e.g., using yesterday's $NO_2$), restricting the model from using unavailable same-day data in live production.

---

## [1.0.0] - Initial Prototype
- **Baseline Model**: Basic PM2.5 prediction using Gradient Boosting Regressors.
- **Temporal Split Validation**: Migrated away from random `train_test_split` to a strict chronological walk-forward split (80/20) to prevent severe time-series data leakage.
