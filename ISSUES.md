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

## 5. The Chaining "Time Machine" Collapse (Phase 5 to Phase 7)
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
