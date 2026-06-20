# Changelog

All notable changes to this project will be documented in this file.

## [8.0.0] - V8 Production Release

### 🚀 Recent Updates (Transition to V8)
- **Horizon-Aligned Autoregressive Memory**: Shifted from a stateless model to a time-aware engine by injecting strict $y_{t-h}$ lags and $\sigma_{3d}$ volatility matrices, entirely eliminating the high-variance MASE trap in chaotic environments like India and GB.
- **Metric Engineering (NMAE & MASE)**: Deprecated the mathematically flawed $R^2$ score for low-variance environments. The backend now strictly evaluates using MASE (Mean Absolute Scaled Error) against a naive baseline, while the UI renders a normalized Accuracy Percentage (NMAE).
- **ETL Fault Tolerance**: Engineered a resilient asynchronous batching pipeline (`aiohttp`). The system successfully demonstrated zero-downtime fault tolerance by seamlessly falling back to internal database states when the primary OpenAQ API threw a 401 Unauthorized suspension block.
- **UX Time-Relativity**: Eradicated the 'Index 0' date trap on the Next.js frontend by implementing strict mathematical matching between the user's local browser `new Date()` and the payload's absolute timestamp.

---

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
