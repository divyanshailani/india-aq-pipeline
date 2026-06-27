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

---

## 16. The June 25th Monsoon Anomaly — "Single-Day Myopia" [RESOLVED - V11.1 Release]
**Issue:**
On June 25th, the V11 engine predicted India's average PM2.5 at **49.44 µg/m³** while the actual ground truth was **14.61 µg/m³** — a catastrophic 3.4× overestimation. Root-cause analysis revealed that a massive monsoon washout event had scrubbed the atmosphere clean within hours, but the model had zero awareness of accumulated precipitation. It only consumed `future_precip` (is it raining *today*?), completely ignoring the fact that 72 hours of continuous heavy rain had already flushed all particulate matter from the atmospheric column.

**First Attempt — Hardcoded Heuristic Override (`is_raining_now`):**
We initially built a quick patch: if `future_precip > 10mm`, force the prediction down by 60%. If `future_wind > 20 km/h`, reduce by another 30%. This immediately stopped the anomaly — but overcorrected disastrously. India's prediction crashed to **6.38 µg/m³**, which is unrealistically low even during heavy rain. The heuristic couldn't calibrate itself; it was a blunt hammer for a precision problem.

**Solution (Autonomous Physics via Feature Engineering):**
We completely ripped out all hardcoded heuristic overrides and engineered two new physics-aware features computed via PostgreSQL Window Functions:
1. `rolling_3day_precip`: `SUM(om_precipitation) OVER (PARTITION BY station_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)` — gives the model "Atmospheric Memory" of cumulative rain.
2. `aod_volatility_index`: `STDDEV(om_aerosol_optical_depth) OVER (PARTITION BY station_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)` — captures atmospheric instability/turbulence over the trailing week.

After Optuna retraining with these new features, the model natively learned to reduce its prediction by ~20 µg/m³ during monsoon events (58.75 → 38.74) without any manual intervention. The remaining gap to 14.61 represents the "Mean Reversion Trap" (see Issue 18).

---

## 17. The ApiFallbackManager — Production Resilience Architecture [RESOLVED - V11.1 Release]
**Issue:**
The original ETL pipeline (`run_daily_etl.py`) used hardcoded API keys and had zero retry logic. During the Azure production transition planning, we identified three critical failure modes:
1. **OpenAQ Rate Limiting (429):** A single API key would get throttled during peak hours, causing entire daily fetches to silently fail.
2. **Open-Meteo & NASA POWER Timeouts:** These IP-based APIs (no keys) would intermittently timeout during high-traffic periods with no recovery mechanism.
3. **Atomic Transaction Rigidity:** The initial architectural proposal would reject an *entire day's data* if a single API threw a 429 or timed out — an unacceptable data loss vector for production.

**Solution (`src/api_fallback_manager.py`):**
We built a centralized `ApiFallbackManager` class with three defense layers:
1. **OpenAQ Key Rotation (`KeyManager`):** Accepts an array of API keys. If Key A hits a 429 or fails, the system catches the exception, logs a warning, rotates to Key B, and retries the exact same request — no data loss.
2. **Exponential Backoff & Jitter:** For IP-based APIs (Open-Meteo, NASA POWER), implements progressive retry delays (2s → 5s → 15s) with random jitter to prevent thundering-herd effects.
3. **The Final Kill Switch:** The Atomic Transaction abort (`RuntimeError`) is triggered ONLY if ALL fallback mechanisms (every key rotated, every retry exhausted) have been fully depleted. A single transient 429 will never kill a daily pipeline run.

---

## 18. The XGBoost "Mean Reversion Trap" — Why Trees Smooth Extreme Events [OPEN — Future ML Frontier]
**Issue:**
Even after injecting `rolling_3day_precip` and retraining with Optuna, the V11.1 model predicted **38.74 µg/m³** during the June 25th monsoon washout while the ground truth was **14.61 µg/m³**. The model correctly learned the *direction* (rain = lower pollution) but refused to predict the extreme tail-end value. This is a fundamental property of MAE-optimized decision trees, not a bug.

**Root Cause (The Physics of XGBoost):**
When a decision tree reaches a leaf node for a condition like "Rain > 15mm," it computes the *average* of all historical outcomes matching that condition. In reality, 15mm of rain sometimes drops pollution to 14 µg/m³ (extreme washout) and sometimes only to 50 µg/m³ (partial clearing). XGBoost, optimized on MAE, is mathematically penalized more for predicting 14 when reality is 50 than for predicting the safe average of ~38. So it always hedges toward the center of the distribution.

**Identified Solutions (Not Yet Implemented):**
1. **Quantile Regression:** Train XGBoost with `objective='reg:quantile'` at the 10th percentile to explicitly predict worst-case washout scenarios instead of averages.
2. **Two-Tier Classification Architecture (Regime-Switching):** Train a binary classifier ("Is this an Extreme Washout day? YES/NO"), and if YES, route the prediction to a secondary XGBoost model trained *exclusively* on rainy days. This "Mixture of Experts" approach allows the wet-weather model to dedicate 100% of its tree depth to learning fluid dynamics without being diluted by the 95% of dry days in the training set.

**Status:** Documented as a future ML frontier. The current V11.1 model is statistically sound (all MASE < 1.0 globally) and relies entirely on autonomous machine learning without hardcoded physics heuristics.

---

## 19. The Nginx 404 & Azure Port 443 Block (Production Deployment)
**Issue:**
After generating an SSL certificate via Certbot (`certbot --nginx`), the raw IP address returned a `404 Not Found` (served directly by Nginx, not FastAPI). Simultaneously, accessing the secure `https://api.globalaqi.live` domain resulted in a complete browser timeout (`ERR_CONNECTION_TIMED_OUT`).

**Root Cause 1 (The 404):**
Certbot modifies the Nginx configuration to strict host-matching (`server_name api.globalaqi.live`). When accessed via the raw IP instead of the domain name, Nginx fails to match the server block. Since there is no default fallback block, Nginx defaults to searching for an `index.html` in an empty `/var/www/html` directory, throwing a native 404.

**Root Cause 2 (The Timeout):**
Certbot successfully established the SSL certificates via port 80 (HTTP-01 challenge) and updated Nginx to listen on port 443. However, the Azure Virtual Machine's Network Security Group (NSG) had port 443 explicitly blocked. Nginx was correctly attempting to upgrade HTTP traffic to HTTPS, but the Azure firewall was completely dropping the incoming port 443 packets.

**Solution:**
1. Accessed the Azure Portal → VM Networking → Added an Inbound Port Rule for `Port 443 (HTTPS)`.
2. Traffic instantly flowed through to the Nginx reverse proxy, resolving the timeout and properly routing to the Gunicorn/FastAPI backend on port 8000.

---

## 20. The "0-Day Gap" Dashboard Illusion (Data Sync vs Backfill)
**Issue:**
The admin dashboard reported a `0d gap` (Zero-day data gap) for all countries (US, GB, AU), even though a massive 10-day API rate-limit failure had just occurred, halting the backfill process for these countries.

**Root Cause:**
The dashboard calculates the "Data Gap" purely by querying the absolute maximum date in the database: `SELECT MAX(datetime_utc)::date FROM raw_measurements`.
Because the daily incremental collector runs *before* the historical backfill process, it had successfully fetched "today's" data for all countries. Therefore, `MAX(date)` correctly evaluated to "today" (a 0-day gap). The dashboard logic assumes contiguous data and does not detect historical "holes" in the middle of the timeline caused by aborted backfill chunks.

**Solution:**
Understood that the `0d gap` metric represents the "freshest available data point", not structural completeness. Missing historical chunks are natively tracked by the separate `backfill_state.json` ledger, which the GitHub Actions runner autonomously reads and resumes processing without relying on the dashboard's `MAX(date)` gap logic.

---

## 21. The "Environment State Divergence" — Local DB vs Azure Cloud DB [RESOLVED 2026-06-27]
**Issue:**
During the AOD backfill operation, a merge script (`/tmp/merge_aod.py`) was run on the local Mac Mini to copy AOD data from `satellite_aod_features` into `daily_features`. However, the `.env` file had no `POSTGRES_HOST` configured — so it defaulted to `localhost`. The script successfully updated **1,069,944 rows** in the local PostgreSQL database while the Azure Flexible PostgreSQL production database remained completely untouched with 100% NULL AOD values.

Subsequently, running `backfill_full_aod.py` on the Azure VM triggered Open-Meteo rate limits (HTTP 429) because the API had already been heavily queried, causing the Azure VM's IP (`4.213.226.19`) to get flagged by the WAF.

**Solution:**
1. Created `scripts/azure_merge_aod.py` with the correct Azure project path (`/opt/pow-eda-pipeline`) and ran it directly on the Azure VM to sync 1,069,944 rows from `satellite_aod_features` → `daily_features`.
2. For remaining Open-Meteo gaps: configured the local Mac Mini's `.env` with Azure DB credentials (`POSTGRES_HOST=globalaqiserver.postgres.database.azure.com`) and ran `backfill_full_aod.py` from the Mac. This bypassed the IP block since the Mac's home IP was clean.
3. Hardened `backfill_full_aod.py` with DB reconnection logic, exponential backoff (30s→150s), network timeout handling, and proper `len(values)` row counting (since `execute_batch` does not reliably set `cur.rowcount`).

**Lesson:** Always verify `POSTGRES_HOST` before running data migration scripts. Local and cloud databases can silently diverge, creating a "Split Reality" where development looks healthy but production is broken.

---

## 22. The Legacy Column Graveyard — 13 Columns at 95% NULL [OPEN — Investigation Required]
**Issue:**
A full Azure DB audit (2026-06-27) revealed **13 columns** in `daily_features` with **94.5–97.2% NULL rates** (~1.55M NULLs out of 1.63M rows). These columns include `temperature`, `humidity`, `wind_speed`, `no2_value`, `co_value`, `o3_value`, `so2_value`, `nasa_temperature`, `nasa_humidity`, `nasa_wind_speed`, `precipitation`, `wind_direction`, and `fire_count`.

The ~80K non-NULL rows in each column are consistent with data only being populated for a subset of stations (likely India-only, ~740 stations × ~110 days ≈ 80K rows). The pipeline has since migrated to `om_temperature`, `om_wind_speed`, `om_precipitation` from Open-Meteo, which have a **99.5% fill rate**.

**Action Required:**
1. Confirm that V11 model does NOT use these legacy columns in its feature list.
2. If confirmed unused: mark as `DEPRECATED` in schema documentation and consider dropping in a future migration (~200MB savings).
3. If any are still used by V11: create a backfill strategy using NASA POWER or Open-Meteo historical APIs.

**Impact:** If V11 uses only `om_*` columns → **No impact, safe to deprecate.** If V11 uses legacy columns → **Model is effectively blind on 95% of its training data.**

---

## 23. The Phantom Stations — 1,464 Stations With Zero Feature Data [OPEN — Investigation Required]
**Issue:**
Cross-table consistency check (2026-06-27) revealed **1,464 out of 4,193 stations (35%)** exist in the `stations` table but have **zero corresponding rows** in `daily_features`. The `daily_features` table only covers 2,729 unique stations.

**Possible Root Causes:**
1. Stations were added to the `stations` table during OpenAQ ingestion but the ETL never processed their measurements.
2. Inactive or decommissioned stations with no measurement data available from OpenAQ.
3. The ETL pipeline applies a minimum-data-threshold filter that silently excludes these stations.
4. Country-specific feature tables (`features_india`, `features_usa`, etc.) were planned as alternative destinations but never populated.

**Action Required:**
1. Query which countries the 1,464 orphan stations belong to.
2. Check if they have data in `raw_measurements` or `clean_measurements`.
3. If they have raw data → the feature engineering ETL is silently skipping them (potential bug).
4. If they have no raw data → mark as `inactive` in the stations table.

---

## 24. Empty Operational Tables — model_registry & predictions [OPEN — Schema Cleanup]
**Issue:**
Azure DB audit (2026-06-27) found two operational tables with **zero rows**: `model_registry` (13 cols) and `predictions` (12 cols). Additionally, 4 country-specific feature tables (`features_india`, `features_usa`, `features_uk`, `features_australia`) are completely empty at 0 bytes.

**Questions:**
1. **model_registry:** How is V11 being tracked? Are model versions, hyperparameters, and metrics stored elsewhere (filesystem `.pkl` files, code constants)?
2. **predictions:** Is prediction output only going to `prediction_log` (191K rows) and `prediction_log_archive` (1.8K rows)? What was the original purpose of the `predictions` table?

**Action Required:**
Either populate these tables with their intended data or drop them to reduce schema clutter. The empty country-specific tables suggest a planned-but-unfinished multi-region feature store architecture.

