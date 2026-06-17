# 🌍 Global AQ Intelligence — Full-Stack Air Quality Prediction System

> **End-to-end ML pipeline + live forecasting site:** OpenAQ ingestion → NASA weather fusion → GradientBoosting models → 30-day PM2.5 predictions → static Next.js site auto-deployed on Vercel.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)](https://nextjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🗺️ Project Journey — From India to Global

This repo didn't start as a global system. It started as a **single-country EDA project** and evolved organically through 5 model versions over ~6 months. Here's the full story:

### Phase 1 — India EDA (v1) · *The Beginning*
Started with the simplest question: *can I predict India's PM2.5 from public data?*
- Pulled 726 Indian stations from OpenAQ API
- Built a 5-phase data cleaning pipeline (NaN removal → outlier capping → dtype fixing)
- Ran EDA across 478 active stations — found **massive seasonal spikes** in Oct-Nov (stubble burning)
- First model: `GradientBoosting v1` with just weather features → **R² = 0.31** — terrible, but a start

**Key problem hit:** `train_test_split` on time-series = data leakage. Random shuffling lets the model see future data during training. Switched to **chronological split** — R² dropped honestly to 0.31. The model wasn't as good as we thought.

### Phase 2 — Cross-Pollutant Features (v2) · *Adding Chemical Context*
Added co-pollutant features: NO₂, CO, O₃, SO₂ as cross-features.
- Model: `gb_pm25_v2_pollutants.pkl`
- R² improved to ~0.48 but still unstable
- **Key learning:** Same-day pollutants are cheating — in production you don't have today's NO₂ to predict today's PM2.5. Switched to lagged (yesterday's) values only.

### Phase 3 — NASA Satellite Weather Fusion (v3) · *The Real Data Source*
Tried Open-Meteo for weather → **63% null rate** for Indian coverage. Unusable.
Switched to **NASA POWER API** — satellite-derived, global, **0.15% nulls**.
- 15 features: weather (temp, humidity, wind, precip, wind direction) + lagged pollutants
- R² = **0.71** — honest, production-safe, no leakage
- Added **NASA FIRMS fire detection** — VIIRS satellite fire points within 100km radius of each station. Stubble burning season (Oct-Nov) signal captured.

### Phase 4 — Temporal Memory Breakthrough (v4) · *The Insight That Changed Everything*
The experiment that defined this project:

| Model | Without Lag | With Lag | Improvement |
|-------|------------|----------|-------------|
| Linear Regression | -0.07 | 1.00* | +1.07 |
| Random Forest | -0.37 | 0.96 | +1.33 |
| **Gradient Boosting** | **-0.32** | **0.97** | **+1.29** |

> **Without temporal lag features, ALL models perform WORSE than the mean baseline (negative R²).** Air quality is a time-series problem. Yesterday's PM2.5 carries 76% of predictive signal. Adding lag_1, lag_7, and rolling means → **R² 0.97, MAE 1.9 µg/m³**.

*Note: LR R²=1.00 flagged as potential `roll_3_mean` leakage — GBM 0.97 is the validated result.*

### Phase 5 — Global Expansion + 30-Day Chained Forecast (v5, current) · *The Public Product*

The model went global: **India 🇮🇳 + USA 🇺🇸 + UK 🇬🇧 + Australia 🇦🇺**

Per-country models trained separately because PM2.5 dynamics differ fundamentally:
- **India:** High pollution, monsoon-driven, fire season — rich seasonal signal
- **USA:** Low baseline, AQI-regulated, wildfire events (West Coast)
- **UK:** Clean maritime air, fragmented community sensors, limited historical data
- **Australia:** Very low PM2.5, NSW EPA reference stations, bushfire spikes

**New prediction engine:** 30-day chained forecast — each predicted day feeds into the next as synthetic lag features. Accuracy degrades gracefully:
- Days 1-7: Direct prediction from real lags → **highest accuracy**
- Days 8-15: First-generation chained → **moderate uncertainty**
- Days 16-30: Long-range trend only → **directional, treat as estimate**

**Confidence tier system per country:**

| Country | R² | MAE | Confidence Tier | Reason |
|---------|-----|-----|----------------|--------|
| 🇺🇸 USA | 0.80 | 1.7 µg/m³ | ✅ High Confidence | EPA AQS reference-grade, 1.4M rows |
| 🇮🇳 India | 0.77 | 8.5 µg/m³ | ✅ High Confidence | CPCB + NASA fire, 31K features |
| 🇦🇺 Australia | 0.64 | 1.6 µg/m³ | 🔵 Low Variance / Stable | Clean air country, NSW EPA |
| 🇬🇧 UK | 0.48 | 2.0 µg/m³ | 🟡 Experimental | 6mo DEFRA data, fragmented sensors |

**Pair with a public Next.js frontend** on Vercel — see [`global-aq-intelligence`](https://github.com/divyanshailani/global-aq-intelligence) *(frontend repo)*.

---

## 🏛️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   OpenAQ     │  │  NASA POWER  │  │ NASA FIRMS │  │  DEFRA   │ │
│  │ 4 countries  │  │  Satellite   │  │  Wildfire  │  │ NSW EPA  │ │
│  │ 200+ stations│  │  Weather     │  │ Detection  │  │  CPCB    │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  └────┬─────┘ │
└─────────┼─────────────────┼────────────────┼───────────────┼───────┘
          │                 │                │               │
          ▼                 ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ETL PIPELINE (local Mac)                      │
│                                                                     │
│  raw_measurements (7.7M rows)                                       │
│       ↓ 5-Phase Cleaning                                            │
│  clean_measurements (7.5M rows)                                     │
│       ↓ Feature Engineering (lag + weather + fire + temporal)       │
│  daily_features (57K rows, 18 features)                             │
│       ↓ Per-country XGBoost/GBM Training                            │
│  models/v5/{IN,US,GB,AU}_pm25_gbr.pkl                               │
│       ↓ 30-day chained prediction                                   │
│  data/site_data/predictions_{IN,US,GB,AU}.json                      │
└─────────────────────────────────────────────────────────────────────┘
          │
          │  git commit + push
          ▼
┌────────────────────────────────┐
│   GitHub: india-aq-pipeline    │  ← this repo (ML backend)
│   /data/site_data/*.json       │
└────────────────┬───────────────┘
                 │
                 │  submodule / file copy
                 ▼
┌────────────────────────────────┐     ┌─────────────────────────────┐
│  global-aq-intelligence        │────▶│  Vercel (auto-deploy)       │
│  Next.js 15 frontend           │     │  Static site rebuilt ~30s   │
│  /public/data/*.json           │     │  No server, no DB           │
└────────────────────────────────┘     └─────────────────────────────┘
```

---

## 📊 Data Flow Diagram

```
[OpenAQ REST API]
    │ /measurements?country=IN&parameter=pm25&limit=1000
    ▼
[fetch_openaq.py] ──── checkpointing (saves last_date per country)
    │
    ▼
[PostgreSQL: raw_measurements]
    7.7M rows | station_id, datetime_utc, value, parameter
    │
    ▼
[run_daily_etl.py] orchestrates →
    │
    ├── [cleaning.py]
    │     Phase 1: Drop nulls (station_id, value, datetime)
    │     Phase 2: Replace placeholder values (-999, 9999, 0 where impossible)
    │     Phase 3: Filter negatives (PM2.5 ≥ 0)
    │     Phase 4: Cap outliers (IQR × 3.5, per-station)
    │     Phase 5: Dtype coercion (float32, UTC timestamps)
    │
    ├── [features.py]
    │     Lag features: lag_1, lag_2, lag_3, lag_7 (shift by N days)
    │     Rolling: roll_3_mean, roll_7_mean (shifted to avoid leakage)
    │     Temporal: month, day_of_week, is_weekend, day_of_year
    │     Spatial: latitude, longitude
    │     Pollutant lags: lag_1_no2, lag_1_co, lag_1_o3, lag_1_so2
    │
    ├── [fetch_nasa_power.py] → weather per country lat/lon centroid
    │     T2M (temp), RH2M (humidity), WS10M (wind speed)
    │
    ├── [fetch_nasa_power_extra.py]
    │     PRECTOTCORR (precipitation), WD10M (wind direction)
    │
    └── [process_firms_fire.py]
          VIIRS fire points → count within 100km radius per station/day

[train_v5.py] ─── per-country training
    │  Chronological 80/20 split
    │  GradientBoostingRegressor(n_estimators=200, max_depth=5)
    │  Evaluation: R², MAE, RMSE on held-out future data
    ▼
[models/v5/{country}_pm25_gbr.pkl]

[predict_pipeline.py] ─── 30-day chained forecast
    │  Day 1-7: real lag features from DB → direct prediction
    │  Day 8-30: predicted value used as synthetic lag_1 for next day
    │  Confidence bands: ±(std of training residuals × horizon_factor)
    ▼
[data/site_data/predictions_{IN,US,GB,AU}.json]
[data/site_data/model_meta.json]
[data/site_data/accuracy.json]
    │
    └── copied to /global-aq-intelligence/public/data/
        → git commit → Vercel rebuild → live site updated
```

---

## 🧠 Model Evolution — v1 → v5

```
v1 (baseline)          v2 (pollutants)       v3 (NASA weather)
R² = 0.31              R² = 0.48             R² = 0.71
MAE = 28.4 µg/m³       MAE = 22.1 µg/m³      MAE = 17.0 µg/m³
Features: 8            Features: 12           Features: 15
India only             India only             India only
Random split ❌         Chrono split ✅         Chrono split ✅
                                              NASA POWER ✅
                                              FIRMS fire ✅

v4 (memory breakthrough)      v5 (global, current)
R² = 0.97                     R² = 0.48–0.80 (per country)
MAE = 1.9 µg/m³               MAE = 1.55–8.52 µg/m³
Features: 18                  Features: 18 per country model
India only                    4 countries ✅
lag_1, lag_7 added ✅          30-day chained forecast ✅
roll_3_mean added ✅           Confidence tiers ✅
                              Public Vercel site ✅
```

---

## 🚧 Problems Faced & How We Fixed Them

### 1. Data Leakage (Critical Bug)
**Problem:** Used `train_test_split(shuffle=True)` → model saw future data during training → fake R² = 0.95.

**Fix:** Switched to strict chronological split. Set cutoff date, everything before = train, after = test. R² dropped to honest 0.31 on v1. Painful but correct.

### 2. Open-Meteo Weather — 63% Nulls for India
**Problem:** Open-Meteo (free weather API) had 63% missing values for Indian stations. Model couldn't learn weather signal.

**Fix:** Switched to **NASA POWER API** (satellite-derived). 0.15% nulls globally. Zero API key required. Fixed by one endpoint change.

### 3. Negative R² Without Lag Features
**Problem:** All models (including Random Forest) had negative R² when predicting PM2.5 without temporal features. Worse than predicting the mean.

**Fix:** Added lag_1 (yesterday's PM2.5), lag_7, rolling means. R² went from -0.32 to 0.97. **Air quality is a memory problem, not a regression problem.**

### 4. UK Model — Fragmented Sensors, Low R²
**Problem:** UK OpenAQ data mixes reference stations with low-cost community sensors. Wildly different accuracy levels. Only 6 months of usable DEFRA data collected.

**Fix:** Flagged UK as "Experimental" tier on the frontend with honest R² = 0.48. Not hidden, not inflated. Added explicit warning on the UK country card. Will improve with more data.

### 5. 30-Day Forecast Error Accumulation
**Problem:** Chained forecasting compounds errors — each predicted day's error becomes the next day's input noise.

**Fix:** Added confidence bands that widen proportionally with horizon (±σ × √horizon). Days 1-7 shown as "direct" (tightest bands), days 16-30 shown as "directional trend only" with explicit label on the frontend.

### 6. Static Site Staleness
**Problem:** Vercel static site shows the same predictions forever unless manually updated. No live fetch on a static export.

**Status:** Solved by local admin pipeline — run daily, commit JSON files, Vercel auto-deploys. GitHub Actions cron job planned for automation.

---

## 📐 Feature Importance (v4)

| Rank | Feature | Importance | Category |
|------|---------|-----------|---------|
| 1 | `lag_1` (yesterday's PM2.5) | 0.76 | Temporal Memory |
| 2 | `lag_7` (week-ago PM2.5) | 0.08 | Temporal Memory |
| 3 | `roll_7_mean` | 0.04 | Temporal Memory |
| 4 | `humidity` | 0.03 | NASA Weather |
| 5 | `wind_speed` | 0.02 | NASA Weather |
| 6 | `month` | 0.02 | Temporal |
| 7 | `precipitation` | 0.01 | NASA Weather |
| 8 | `fire_count_lag_1` | 0.004 | NASA FIRMS |
| 9-18 | pollutant lags, spatial | 0.004 | Other |

---

## 📡 Data Sources

| Source | API | Data | Update Lag | Key Required |
|--------|-----|------|-----------|--------------|
| **OpenAQ** | REST v3 | PM2.5, PM10, NO₂, CO, O₃ from 2700+ stations | ~12-24hr | Yes (free) |
| **NASA POWER** | Daily Point API | Temperature, Humidity, Wind, Precipitation | Same-day | No |
| **NASA FIRMS** | VIIRS NRT | Active fire detections (100km radius) | ~3hr | No (basic) |
| **DEFRA AURN** | UK-AIR API | UK reference monitoring network | ~1hr | No |
| **NSW EPA** | AirWatch | Australian state EPA reference | ~1hr | No |
| **EPA AQS** | *(not used)* | Historical only, 6+ month lag | 6mo+ | Yes |
| **AirNow** | REST | US real-time AQI | ~1hr | Yes (free) |

---

## 🏗️ Project Structure

```
global-aq-intelligence-pipeline/         (was: pow-eda-pipeline / india-aq-pipeline)
│
├── scripts/                             # Production pipeline scripts
│   ├── fetch_openaq.py                  # Multi-country OpenAQ ingestion
│   ├── fetch_nasa_power.py              # NASA POWER: temp/humidity/wind
│   ├── fetch_nasa_power_extra.py        # NASA POWER: precip/wind dir
│   ├── fetch_firms_fire.py              # NASA FIRMS fire API
│   ├── process_firms_fire.py            # Fire points → regional counts
│   ├── fetch_defra_bulk.py              # UK DEFRA bulk historical
│   ├── fetch_epa_bulk.py                # US EPA bulk historical
│   ├── fetch_nsw_bulk.py                # Australia NSW EPA bulk
│   ├── run_daily_etl.py                 # Orchestrator: clean → features
│   ├── train_v5.py                      # v5 per-country model training
│   ├── predict_pipeline.py              # 30-day chained forecast engine
│   ├── build_global_features.py         # Global feature matrix builder
│   ├── admin_dashboard.py               # FastAPI local admin panel
│   └── auto_collect.py                  # macOS launchd wrapper
│
├── src/                                 # Reusable Python modules
│   ├── cleaning.py                      # 5-phase cleaning pipeline
│   ├── features.py                      # Feature engineering module
│   ├── aggregations.py                  # Station → country aggregation
│   └── process_aq.py                    # AQ data processing utilities
│
├── notebooks/                           # Research & exploration
│   ├── 01_indian_aq_clean.ipynb         # Phase 1: 5-phase cleaning
│   ├── 02_eda.ipynb                     # Phase 2: EDA, distributions
│   ├── 03_feature_engineering.ipynb     # Phase 3: feature design
│   ├── 05_ml_model_clean.ipynb          # Phase 4: v1-v4 model training
│   ├── global_ml_model.ipynb            # Phase 5: global v5 training
│   └── lstm_pm25.ipynb                  # LSTM experiment (shelved)
│
├── models/                              # Trained models (gitignored)
│   ├── v5/
│   │   ├── IN_pm25_gbr.pkl              # India model (R²=0.77)
│   │   ├── US_pm25_gbr.pkl              # USA model  (R²=0.80)
│   │   ├── GB_pm25_gbr.pkl              # UK model   (R²=0.48)
│   │   ├── AU_pm25_gbr.pkl              # Australia  (R²=0.64)
│   │   └── all_models_meta.json
│   └── [older versions v1-v4]
│
├── data/
│   ├── site_data/                       # JSON for Vercel frontend
│   │   ├── predictions_IN.json
│   │   ├── predictions_US.json
│   │   ├── predictions_GB.json
│   │   ├── predictions_AU.json
│   │   ├── model_meta.json
│   │   └── accuracy.json
│   ├── weather_nasa_*.csv               # Cached NASA weather (gitignored)
│   └── raw/                             # Raw API data (gitignored)
│
├── plots/                               # Research visualizations
│   ├── forecast_horizons.png
│   ├── model_comparison_memory.png
│   └── v4_feature_importance.png
│
├── reports/                             # EDA reports
│   ├── monthly_pm25_trends.png
│   ├── seasonal_comparison.png
│   └── pollutant_correlation.png
│
├── sql/
│   └── schema.sql                       # PostgreSQL schema
│
├── logs/                                # Runtime logs (gitignored)
├── .env.example                         # API key template
├── .gitignore                           # Excludes models, raw data, keys
└── README.md
```

---

## 🚀 Setup

```bash
# Clone
git clone https://github.com/divyanshailani/global-aq-intelligence-pipeline.git
cd global-aq-intelligence-pipeline

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn \
            psycopg2-binary requests joblib fastapi uvicorn

# Copy env template and add your keys
cp .env.example .env
# edit .env → add OPENAQ_API_KEY, AIRNOW_API_KEY

# PostgreSQL setup
psql -U postgres -f sql/schema.sql

# Fetch data (multi-country)
python scripts/fetch_openaq.py --country IN
python scripts/fetch_openaq.py --country US
python scripts/fetch_openaq.py --country GB
python scripts/fetch_openaq.py --country AU

# Fetch NASA weather (no key required)
python scripts/fetch_nasa_power.py
python scripts/fetch_nasa_power_extra.py

# Run ETL pipeline
python scripts/run_daily_etl.py

# Train v5 models
python scripts/train_v5.py

# Generate 30-day predictions
python scripts/predict_pipeline.py

# Start admin dashboard
uvicorn scripts.admin_dashboard:app --port 8001 --reload
```

---

## 🔐 Security

```bash
# .gitignore enforces:
*.pkl               # No model files (large, private)
*.csv               # No raw data
data/raw/           # No fetched API data
.env*               # No API keys ever
logs/               # No runtime logs
__pycache__/        # No Python cache
```

**Never committed:**
- API keys (OpenAQ, AirNow, Visual Crossing)
- Model `.pkl` files (stored locally, share via DVC if needed)
- Raw measurement CSVs
- PostgreSQL credentials

---

## 🔭 Roadmap

- [x] India EDA — 5-phase cleaning (v1)
- [x] Cross-pollutant features (v2)
- [x] NASA POWER weather fusion, FIRMS fire (v3)
- [x] Temporal memory breakthrough — lag features (v4) R²=0.97
- [x] Global expansion — 4 countries (v5)
- [x] 30-day chained forecast with confidence bands
- [x] Confidence tier system per country
- [x] Public Vercel frontend (Next.js 15)
- [x] AQI-themed animated canvas background
- [ ] Admin panel: one-click fetch → predict → deploy
- [ ] Daily GitHub Actions cron (auto-update Vercel)
- [ ] Weekly model retrain with R² guard
- [ ] Live validation log: predicted vs actual
- [ ] LSTM/Transformer experiment (R² > 0.98 target)
- [ ] NASA FIRMS integration for AU/US wildfire events

---

## 🔗 Related Repos

| Repo | Purpose |
|------|---------|
| [`global-aq-intelligence`](https://github.com/divyanshailani/global-aq-intelligence) | Next.js frontend (Vercel) |
| [`india-aq-pipeline`](https://github.com/divyanshailani/india-aq-pipeline) | ← this repo (renamed from `india-aq-pipeline`) |

---

## 👤 Author

**Divyansh Ailani** — [GitHub](https://github.com/divyanshailani) · [LinkedIn](https://www.linkedin.com/in/divyansh-ailani-225925380/)

*Simulation Architect | First-Principles Engineering | BCA*

> *"R² = 0.71 is honest. R² = 0.97 with temporal memory is the same model — with the right features."*
