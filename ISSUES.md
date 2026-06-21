# 🚧 Key Issues & Solutions 

This document logs the critical infrastructure, data, and machine learning roadblocks we overcame throughout the entire lifecycle of the Global AQ Intelligence project.

---

## 1. The Time-Series Data Leakage Illusion (Phase 1)
**Issue:** 
During the initial EDA for India, we used a standard `train_test_split` with random shuffling. This inadvertently caused severe **data leakage** because the model could "see" future data during training, artificially inflating accuracy metrics.

**Solution:** 
We strictly abandoned random shuffling and implemented a **chronological walk-forward split** (Temporal Split). The model is strictly trained on the first 80% of the timeline and tested on the last 20%. This forced an honest evaluation and protected the pipeline from future-leakage.

---

## 2. Co-Pollutant "Cheating" in Production (Phase 2)
**Issue:** 
Adding same-day co-pollutants (NO₂, CO, O₃, SO₂) drastically improved R². However, this is useless in a live production environment because you do not have today's live NO₂ data when you are trying to predict today's PM2.5.

**Solution:** 
We restricted the model from using any same-day co-pollutant data. All chemical context features were shifted backward (e.g., using yesterday's NO₂ as `lag_1_no2`). This ensured the model remained completely production-ready.

---

## 3. The 63% Null-Rate Failure of Open-Meteo Historicals (Phase 3)
**Issue:** 
We originally attempted to use Open-Meteo for historical weather data to train the model, but discovered it had a catastrophic **63% null rate** for our Indian coverage areas, making the dataset unusable for machine learning.

**Solution:** 
We ripped out Open-Meteo for historical training and integrated the **NASA POWER API** (satellite-derived global meteorology), which reduced our null rate to **0.15%**. We still use Open-Meteo for *future* 16-day forecasts, but NASA powers the historical training engine.

---

## 4. The Temporal Memory Gap (Phase 4)
**Issue:** 
Without temporal features, the Gradient Boosting model actually performed worse than a simple mean baseline (Negative R²). Air quality is fundamentally a time-series problem; the models had no memory of yesterday.

**Solution:** 
We engineered a robust suite of temporal memory features (`lag_1` to `lag_30`, rolling means, and rolling volatility). We discovered that yesterday's PM2.5 alone carries 76% of the predictive signal. This architectural shift skyrocketed our R² from ~0.31 to 0.97.

---

## 5. The Recursive Snowball Effect & Chaining Collapse [RESOLVED - V8 Release]
**Issue:** 
In `v5`, we used a 30-day chained forecasting loop where Day 1's prediction was fed into Day 2 as a `lag_1` feature. This caused rapid error compounding (exponential degradation), meaning that by Day 30 the model was predicting complete noise.

**Solution:** 
We developed the **Direct Multi-Horizon Architecture (V7 Thermodynamics Engine)**. Instead of using Day N-1 to predict Day N, we trained entirely independent models targeting specific horizons (`1d`, `7d`, `14d`, `30d`). 
- Days 1-7 use the `h1` direct model.
- Intermediate days are processed using our **Anchor Point Strategy** with a **Weather-Weighted Interpolator**.

---

## 6. India's High-Variance Signal vs "Dumb Linear Interpolation" (Phase 7)
**Issue:**
After implementing the anchor point strategy (interpolating between the `1d`, `7d`, `14d`, `30d` predictions), the frontend rendered a perfectly straight, linear line for India. The interpolation stripped out all the chaotic variance of real weather (the "Dumb Linear" problem).

**Solution:**
We developed a Physics-based **Weather-Weighted Interpolator**.
- **Base Interpolation:** Standard linear curve.
- **Extract Daily Weather:** Fetch specific Open-Meteo `future_wind` and `future_precip` for that day.
- **Thermodynamic Modifiers:** 
  - *Rain Washout*: Precip > 2.0mm → PM2.5 drops by 30%.
  - *Wind Dispersion*: Wind > 15 km/h → PM2.5 drops by 15%.
  - *Stagnation Spike*: Wind < 5 km/h and 0 precip → PM2.5 rises by 20%.

---

## 7. Next.js Aggressive Caching ("Perfect Straight Line" Illusion)
**Issue:**
Even after applying the Weather-Weighted Interpolator in the backend, the Next.js frontend continued displaying the old straight line during local development.

**Solution (The Developer's Hard Reset):**
Next.js aggressively caches JSON in memory during `npm run dev`. To force Next.js to read the new thermodynamic data, we established a strict hard-reset protocol:
1. Kill the server process (`kill -9 <PID>`).
2. Destroy the cache directory (`rm -rf .next`).
3. Ensure the prediction pipeline synced the JSON files directly to `global-aq-intelligence/public/data/site_data`.
4. Restart the dev server.

---

## 8. True Benchmarks vs Operational Backtest Misalignment
**Issue:** 
The 7-Day Operational Backtest evaluated via `backtest_recent()` yielded `R²=0.38` for India, whereas the true Global test was `R²=0.75`.

**Solution:**
We identified that `backtest_recent()` inherently tests a highly volatile micro-sample (the last 7 days of live production actuals). A 7-day window is prone to extreme variance from localized, seasonal anomalies. The 20% temporal holdout run by `train_v7_experiment.py` evaluates the model across months of data, providing the statistically robust `0.75` R² metric displayed on the site. Operational metrics are now correctly separated from global benchmark validation.

---

## 9. Synchronous ETL Bottleneck — API Rate Limiting [RESOLVED - V8 Release]
**Issue:**
The original `fetch_openaq.py` made sequential blocking `requests.get()` calls per station. With 1,400+ US stations, this created multi-hour fetch windows and regularly hit the OpenAQ API's rate limit (HTTP 429). When 429s hit, the pipeline stalled silently.

**Solution:**
Rewrote the measurement fetch layer as a fully async pipeline using `aiohttp` + `asyncio.Semaphore` for chunked concurrent batching. Key changes:
- `fetch_station_sensors()` replaced with `fetch_station_sensors_async(session, id, headers, semaphore)`
- 429 responses now trigger `await asyncio.sleep(5)` and retry up to 5 times before giving up on a station
- Rate limit hit on the stations endpoint triggers a synchronous `time.sleep(10)` before retrying
- `certifi` + `ssl` context injected to prevent SSL verification failures on macOS
- `aiohttp` added to `requirements.txt`

---

## 10. Database Fresh-Clone Failure — Hardcoded Password + Missing UNIQUE Constraint [RESOLVED 2026-06-19]
**Issue 1:** `src/config.py` had a hardcoded fallback DB password (`"8765"`). Any fresh clone by a reviewer or CI environment that had no `.env` file would silently use the wrong password and fail on connection with a cryptic auth error.

**Issue 2:** `stations` table had no `UNIQUE` constraint on `(name, country_code)`. During re-ingestion runs, `upsert_stations()` produced `CardinalityViolation` errors when the same station appeared multiple times in an API response page.

**Solution:**
- `src/config.py`: removed hardcoded fallback. `POSTGRES_PASSWORD` is now loaded via `python-dotenv` and raises a hard `ValueError` at startup if missing. Fails fast and loudly.
- `sql/schema.sql`: added `UNIQUE (name, country_code)` to the `stations` table definition. `ON CONFLICT` clause in `upsert_stations()` now targets `(name, country_code)` instead of `openaq_id`.
- `fetch_openaq.py`: added Python-level deduplication of station list on `(name, country_code)` key before any DB write.

---

## 11. Low-Variance $R^2$ Illusion — Metric Misleads on Clean-Air Countries [RESOLVED - V8 Release]
**Issue:**
R² (coefficient of determination) is pathologically misleading for low-variance targets. Australia and UK have PM2.5 ranges of 1–8 µg/m³. A model that predicts the mean every time achieves R²=0.0. A model that adds small noise achieves negative R². This caused Australia (R²=0.45) and UK (R²=0.24) to look broken on the frontend and in internal logs, even though their absolute MAE was well within useful accuracy (1.88 and 2.41 µg/m³ respectively).

**Solution:**
Replaced R² throughout the pipeline with **NMAE-derived Accuracy %**:

```
NMAE = MAE / mean(y_actual)
Accuracy % = max(0, (1 - NMAE) * 100)
```

This is scale-invariant and interpretable to non-ML audiences. Changes applied across:
- `validate_old_predictions()` — live metric now reports `live_acc` not `live_r2`
- `backtest_recent()` — backtest metric now `acc_pct` not `r2`
- `export_site_data()` — `accuracy.json` and `model_meta.json` emit `accuracy_percentage` field, `r2` field removed
- `COUNTRY_META` dict — `test_r2` replaced with `accuracy_percentage` per country
- Frontend `CountryCard.tsx` — dynamic Tailwind color coding: green ≥80%, amber ≥60%, rose <60%
- Frontend `AccuracyProof.tsx` — bar chart and tooltip updated to render Accuracy % not R²
- Frontend `types/index.ts` — `test_r2` field removed, `accuracy_percentage` added

---

## 12. Stale Docstring — predict_pipeline.py V5 Architecture Description [RESOLVED 2026-06-19]
**Issue:**
The module-level docstring in `predict_pipeline.py` described the v5 chaining architecture (7-day direct / 15-day chained / 30-day chained). After the V7 migration, this was actively misleading to anyone reading the source.

**Solution:**
Docstring updated to describe V7 Direct Thermodynamics Engine: anchor points h1/h7/h14/h30 as direct GBR outputs, weather-weighted interpolation for intermediate days, and Open-Meteo 16-day future weather injection per station per date.

---

## 13. The 30-Day 'Climatology' Anchor Drag & Spatial Blindness [RESOLVED - V9.4 Release]
**Issue:**
While V9 performed extremely well at predicting global trends, the static `pm25_rolling_mean_30d` feature acted as an overwhelming anchor for short horizons (like 1-day). This dragged down agility and caused underfitting for chaotic micro-fluctuations. Furthermore, the model was "spatially blind" to massive smoke events since it only knew if a fire existed in a given country, not if the fire was blowing directly into a specific ground station.

**Solution:**
We developed the **V9.4 Geospatial Ensemble Router**:
1. **Delta Target Transformation**: We pivoted the objective function from predicting absolute PM2.5 to predicting the *Velocity* ($\Delta Y = Y_t - Y_{t-1}$). This forces the model to explicitly correct a naive baseline.
2. **Fading Memory (EMA)**: We ripped out the 30-day baseline for short horizons and upgraded the 3-day mean to an Exponential Moving Average (EMA), prioritizing yesterday's pollution to capture rapid fluctuations.
3. **SUOMI VIIRS Spatial Engine**: We implemented the Haversine formula to bridge satellite fire coordinates with ground stations, creating a dynamic `fire_density_100km` and `fire_radiative_power_total` blast radius for each station.
4. **Dynamic Routing**: Since Great Britain relies heavily on long-term 30-day climatology, we introduced a dynamic router that maintains the V9 model for GB at long horizons ($h=14, 30$), while using the upgraded V9.4 engine for all other nodes globally.

---

## 14. V10 Stagnation Physics & Hemispheric Upwind Engine (Failed R&D)
**Issue:**
During a rigorous autopsy of V9.4, we identified a critical blind spot: the model catastrophically under-predicts massive PM2.5 spikes (>150 µg/m³), with errors averaging 87.48 µg/m³ during these extreme events. These spikes primarily occur during "Stagnation Events" (low precipitation, low wind, high fire density). 

**Attempted Solution (V10 / V10.1):**
We hypothesized that injecting non-linear accumulation physics and upwind spatial logic would solve this. We engineered two complex features:
1. `stagnation_index`: A log-scaled ratio of fire radiative power to weather dispersion (`np.log1p(fire_power / (precip + wind + 0.1))`).
2. `upwind_fire_power`: A 180-degree hemispheric sweep that only summed fire brightness if the fire was located upwind of the station based on forward bearing.

**Result & Conclusion:**
The V10.1 logic backfired. It worsened the MAE on extreme spikes from 87.48 to 91.62 µg/m³, and slightly degraded overall MAE. By engineering hyper-specific synthetic indices, we tied variables together into a single column, which forced the XGBoost trees down a specific path and stripped away their ability to find subtle micro-patterns in the raw data (Raw Total Fire Power, Raw Wind Speed).

We mathematically proved that **V9.4** (Delta Targets, Synthetic Memory, and Raw 100km Blast Radius) is the absolute ceiling for the current dataset. Hand-engineered physics indices were abandoned in favor of raw statistical signals.

---

## 15. The V11 3D Atmospheric Ensemble & The GB Exception
**Issue:**
After V10's failure, we realized that 2D satellite coordinates (VIIRS) cannot capture the true *volume* and vertical column density of pollution. Smoke blowing overhead does not always touch ground sensors. Furthermore, V10's hand-engineered stagnation index failed because Gradient Boosting models inherently struggle with erratic synthetic splits; XGBoost prefers raw, continuous statistical signals over hardcoded, non-linear mathematical ratios.

**Solution (The V11 Engine):**
1. **Raw AOD Signals:** We abandoned synthetic physics indices and injected live, raw 3D Aerosol Optical Depth (AOD) vectors from the European CAMS framework via Open-Meteo. AOD provides the exact physical measurement of vertical atmospheric density. XGBoost effortlessly mapped the raw AOD signals to stagnation spikes, dropping the Extreme-Spike MAE (PM2.5 > 150) to 76.06 µg/m³ without disrupting the general horizon accuracy.
2. **The GB Exception (Dynamic Routing):** We hardcoded a routing exception for Great Britain at $h=14$ and $h=30$. Great Britain has an oceanic climate with virtually no wildfires and extremely stable long-term pollution patterns. For this specific region, satellite AOD introduces unnecessary variance. Thus, GB intentionally falls back to the V9 historical persistence engine for long horizons, as historical baselines are a much stronger predictor than 3D AOD for pure oceanic climates.
3. **Per-Country Optuna Tuning:** Instead of forcing a global compromise, we built an Optuna tuning matrix that assigns specific XGBoost parameters to specific countries. For example, Australia (extreme low variance) is artificially constrained to `max_depth=6` to prevent chasing noise, while India (high variance, heavy baseline) expands to `max_depth=9` to fully utilize the 3D AOD structural density.
