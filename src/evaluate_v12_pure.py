"""
Global AQ Intelligence — V12 Pure Evaluation Engine
=====================================================
First-principles, phase-shift-aligned evaluation of all 16 V12 XGBoost models.

Architectural Rules:
  1. Zero Data Leakage: Fresh .copy() per horizon. Nuclear drop of ALL target_* cols.
  2. Phase-Shift Alignment: Predictions from day t compared to actuals at day t+h.
  3. Honest MASE: Naive baseline = persistence model (PM2.5 at day t → day t+h).
  4. No Artificial Imputation: AOD nulls passed directly to XGBoost hist tree method.
  5. Matplotlib Gap Fix: datetime index + resample('D').asfreq() for honest time gaps.

Outputs:
  - Terminal: Markdown table (Country, Horizon, Evaluable Samples, MAE, NMAE, MASE)
  - plots/v12_pure_eval/overlay_graph.png: Actual vs V12 spike capture
  - plots/v12_pure_eval/error_decay.png: MAE decay across horizons per country

Usage:
    python -m src.evaluate_v12_pure
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "daily_features_full.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "v12"
PLOTS_DIR = PROJECT_ROOT / "plots" / "v12_pure_eval"
HOLDOUT_START = datetime.date(2026, 1, 1)
COUNTRIES = ["AU", "GB", "IN", "US"]
HORIZONS = [1, 7, 14, 30]

# Columns that must NEVER appear in the feature matrix
NUCLEAR_BLACKLIST = {
    "value", "date", "parameter", "country_code", "station_id",
    "pm25_delta_1", "pm25_delta_7",
}


# ============================================================================
# Data Loading
# ============================================================================
def load_parquet() -> pd.DataFrame:
    """Load the master Parquet file. Zero imputation — AOD nulls preserved."""
    print(f"📦  Loading {PARQUET_PATH.name}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"    {len(df):,} rows × {len(df.columns)} cols")
    print(f"    Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"    Stations:   {df['station_id'].nunique():,}")
    return df


# ============================================================================
# Model Loading
# ============================================================================
def load_model(country: str, horizon: int):
    """Load a V12 XGBoost model from native JSON format. Returns None if missing."""
    model_path = MODELS_DIR / country / f"horizon_{horizon}" / "model.json"
    if not model_path.exists():
        return None
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model


# ============================================================================
# Nuclear Drop — Feature Extraction with Leakage Detection
# ============================================================================
def extract_clean_features(model: xgb.XGBRegressor) -> list:
    """Extract model features with triple-layer Nuclear Drop safety.

    Layer 1: Reject any feature containing 'target_' (cascade leakage).
    Layer 2: Reject any feature in the NUCLEAR_BLACKLIST (metadata columns).
    Layer 3: Assert the final feature list is non-empty.

    Raises RuntimeError if contamination is detected.
    """
    raw_features = model.get_booster().feature_names

    # Layer 1 — Target cascade detection
    leaked = [f for f in raw_features if f.startswith("target_")]
    if leaked:
        raise RuntimeError(
            f"🚨 LEAKAGE DETECTED! Model contains target features: {leaked}. "
            f"This model is contaminated and CANNOT be evaluated honestly."
        )

    # Layer 2 — Metadata column exclusion
    clean = [f for f in raw_features if f not in NUCLEAR_BLACKLIST]

    # Layer 3 — Sanity check
    if not clean:
        raise RuntimeError("Feature list is empty after Nuclear Drop. Model is invalid.")

    return clean


# ============================================================================
# Single Country × Horizon Evaluation
# ============================================================================
def evaluate_single(
    df_country: pd.DataFrame,
    country: str,
    horizon: int,
    model: xgb.XGBRegressor,
) -> dict | None:
    """Evaluate a single country × horizon with strict phase-shift alignment.

    Phase-Shift Rule:
        - Features extracted at day t.
        - Model predicts PM2.5 at day t+h.
        - Compared against ACTUAL PM2.5 at day t+h.
        - Naive baseline (persistence): PM2.5 at day t predicts day t+h.

    Anti-Leakage:
        - Fresh .copy() per call — zero mutation risk.
        - ONLY the current horizon's target is created.
        - Nuclear Drop enforced on features.

    No Imputation:
        - AOD nulls are NOT filled. XGBoost hist handles NaN natively.
        - Features with NaN are NOT dropped (unlike the old eval script).
    """
    # ── Fresh copy (anti-mutation) ──────────────────────────────────
    df_h = df_country.copy()
    df_h = df_h.sort_values(["station_id", "date"]).reset_index(drop=True)

    # ── Create ONLY this horizon's target (no cascade) ─────────────
    target_col = f"target_{horizon}d"
    df_h[target_col] = df_h.groupby("station_id")["value"].shift(-horizon)

    # ── Filter to holdout period ───────────────────────────────────
    df_h = df_h[df_h["date"] >= HOLDOUT_START].copy()

    # ── Drop rows where target is NaN (no future actual exists) ────
    df_h = df_h.dropna(subset=[target_col])

    # ── Drop rows where current value is NaN (needed for MASE) ─────
    df_h = df_h.dropna(subset=["value"])

    if len(df_h) == 0:
        return None

    # ── Extract clean features (Nuclear Drop enforced) ─────────────
    features = extract_clean_features(model)

    # Verify all features exist in the dataframe
    missing = [f for f in features if f not in df_h.columns]
    if missing:
        print(f"    ⚠️  Missing features in data: {missing}")
        return None

    # ── Build feature matrix — NO imputation of AOD nulls ──────────
    # XGBoost's hist tree method handles NaN natively via learned
    # optimal split directions. Filling nulls would corrupt the signal.
    X = df_h[features]

    y_true = df_h[target_col].values       # Actual PM2.5 at t+h
    y_current = df_h["value"].values        # Actual PM2.5 at t (persistence baseline)

    # ── Predict ────────────────────────────────────────────────────
    y_pred = model.predict(X)

    # ── Compute Metrics ────────────────────────────────────────────
    # MAE: Mean Absolute Error
    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))

    # NMAE: Normalized MAE (by mean of actuals)
    mean_actual = float(np.mean(y_true))
    nmae = mae / mean_actual if mean_actual > 0 else np.nan

    # MASE: Mean Absolute Scaled Error (honest persistence baseline)
    # Naive prediction = "PM2.5 won't change from today"
    # If MASE < 1.0 → model beats the naive baseline
    naive_errors = np.abs(y_true - y_current)
    naive_mae = float(np.mean(naive_errors))
    mase = mae / naive_mae if naive_mae > 0 else np.nan

    # Accuracy: (1 - NMAE) * 100, floored at 0%
    accuracy = max(0.0, (1.0 - nmae) * 100) if not np.isnan(nmae) else 0.0

    return {
        "country": country,
        "horizon": horizon,
        "horizon_label": f"{horizon}d",
        "samples": len(df_h),
        "mae": mae,
        "nmae": nmae,
        "mase": mase,
        "accuracy": accuracy,
        # Carry evaluation data for plotting (lightweight columns only)
        "_plot_data": df_h[["date", "station_id", "value", target_col]].assign(
            prediction=y_pred
        ),
    }


# ============================================================================
# Plotting — Overlay Graph (Spike Capture)
# ============================================================================
def plot_overlay(results: list, output_path: Path):
    """Plot Actual vs V12 for the most volatile holdout station.

    Matplotlib Gap Fix:
        - date → pd.Timestamp index
        - .resample('D').asfreq() fills temporal gaps with NaN
        - Matplotlib naturally breaks the line at NaN gaps
    """
    # ── Find the best station: high volatility AND enough data ───
    best_station_id = None
    best_score = 0.0
    best_volatility = 0.0
    best_result = None

    for r in results:
        if r is None:
            continue
        if r["horizon"] != 1:
            continue

        plot_data = r["_plot_data"]
        for sid, grp in plot_data.groupby("station_id"):
            n = len(grp)
            if n < 20:
                continue
            vol = grp["value"].std()
            # Composite: volatility weighted by sample coverage
            score = vol * np.sqrt(n)
            if score > best_score:
                best_score = score
                best_volatility = vol
                best_station_id = sid
                best_result = r

    if best_result is None:
        print("    ⚠️  No suitable station found for overlay plot.")
        return

    country = best_result["country"]
    target_col = f"target_{best_result['horizon']}d"
    df_station = best_result["_plot_data"]
    df_station = df_station[df_station["station_id"] == best_station_id].copy()
    n_points = len(df_station)

    # ── Matplotlib Gap Fix ─────────────────────────────────────────
    df_station["date"] = pd.to_datetime(df_station["date"])
    df_station = df_station.set_index("date").sort_index()
    df_station = df_station.resample("D").asfreq()  # NaN for missing days

    # ── Plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        df_station.index, df_station[target_col],
        color="#1976D2", linewidth=2.0, label="Actual (t+1)", zorder=3,
    )
    ax.plot(
        df_station.index, df_station["prediction"],
        color="#E64A19", linewidth=1.8, linestyle="--",
        label="V12 Predicted", zorder=2,
    )

    # Error shading
    ax.fill_between(
        df_station.index,
        df_station[target_col],
        df_station["prediction"],
        alpha=0.12, color="#E64A19", zorder=1,
    )

    ax.set_title(
        f"V12 Phase-Shift Overlay — Station {best_station_id} ({country})\n"
        f"1-Day Forecast  ·  σ = {best_volatility:.1f} µg/m³  ·  "
        f"{n_points} evaluable days",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Overlay graph → {output_path}")


# ============================================================================
# Plotting — Error Decay (MAE vs Horizon)
# ============================================================================
def plot_error_decay(results: list, output_path: Path):
    """Line chart: MAE across 1d → 7d → 14d → 30d per country.

    Physically correct models should show monotonically increasing MAE.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {
        "AU": ("#00ACC1", "o"),
        "GB": ("#43A047", "s"),
        "IN": ("#FB8C00", "D"),
        "US": ("#8E24AA", "^"),
    }

    for cc in COUNTRIES:
        cc_results = sorted(
            [r for r in results if r is not None and r["country"] == cc],
            key=lambda r: r["horizon"],
        )
        if not cc_results:
            continue

        horizons = [r["horizon"] for r in cc_results]
        maes = [r["mae"] for r in cc_results]
        samples = [r["samples"] for r in cc_results]
        color, marker = palette.get(cc, ("#666", "o"))

        ax.plot(
            horizons, maes,
            marker=marker, linewidth=2.5, markersize=9,
            label=cc, color=color, zorder=3,
        )

        # Annotate each point with MAE and sample count
        for h, m, n in zip(horizons, maes, samples):
            sample_warning = " 🔴" if n < 200 else ""
            ax.annotate(
                f"{m:.1f}\n(n={n:,}{sample_warning})",
                (h, m),
                textcoords="offset points",
                xytext=(0, 16),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    ax.set_title(
        "V12 Error Decay — MAE vs Forecast Horizon\n"
        "Physically correct: MAE increases with horizon  ·  "
        "🔴 = < 200 samples (noisy)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
    ax.set_ylabel("MAE (µg/m³)", fontsize=12)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(["1d", "7d", "14d", "30d"], fontsize=11)
    ax.legend(fontsize=11, title="Country", title_fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Error decay chart → {output_path}")


# ============================================================================
# Plotting — 2×2 Per-Country Forecast Grids (V11 GitHub Style)
# ============================================================================
def plot_country_grids(results: list, output_dir: Path):
    """Generate a 2×2 grid per country: Actual vs V12 across all 4 horizons.

    Matches the V11 GitHub graph style:
        - Blue solid line = Actual (future PM2.5 at t+h)
        - Red dashed line = V12 Predicted
        - MAE shown in each subplot title
        - Matplotlib Gap Fix applied

    Picks the station with the most evaluable holdout days per country.
    One PNG saved per country: forecast_grid_{CC}.png
    """
    for cc in COUNTRIES:
        cc_results = [r for r in results if r is not None and r["country"] == cc]
        if not cc_results:
            continue

        # Pick the best station: most data across h=1d (largest holdout coverage)
        h1_result = next((r for r in cc_results if r["horizon"] == 1), None)
        if h1_result is None:
            continue

        # Find station with most data points for h=1d
        best_sid = None
        best_count = 0
        for sid, grp in h1_result["_plot_data"].groupby("station_id"):
            if len(grp) > best_count:
                best_count = len(grp)
                best_sid = sid

        if best_sid is None or best_count < 5:
            continue

        # Build 2×2 figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"V12 (Challenger Pure) Forecast: Station {best_sid} ({cc})",
            fontsize=16, fontweight="bold", y=0.98,
        )

        for idx, h in enumerate(HORIZONS):
            ax = axes[idx // 2][idx % 2]
            h_result = next((r for r in cc_results if r["horizon"] == h), None)

            if h_result is None:
                # No model/data for this horizon
                ax.text(
                    0.5, 0.5, f"No Data for {h}d",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#666",
                )
                ax.set_title(f"{h}-Day Forecast", fontsize=12, fontweight="bold")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.2)
                continue

            target_col = f"target_{h}d"
            df_station = h_result["_plot_data"]
            df_station = df_station[df_station["station_id"] == best_sid].copy()

            if len(df_station) < 2:
                ax.text(
                    0.5, 0.5, f"No Station Data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#666",
                )
                ax.set_title(f"{h}-Day Forecast", fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.2)
                continue

            # Matplotlib Gap Fix — ensure unique date index per station
            df_station["date"] = pd.to_datetime(df_station["date"])
            # Some stations may have duplicate date entries; aggregate first
            df_station = df_station.groupby("date").first()
            df_station = df_station.resample("D").asfreq()

            # Compute per-subplot MAE
            valid_mask = df_station[target_col].notna() & df_station["prediction"].notna()
            if valid_mask.sum() > 0:
                subplot_mae = np.mean(
                    np.abs(
                        df_station.loc[valid_mask, target_col].values
                        - df_station.loc[valid_mask, "prediction"].values
                    )
                )
            else:
                subplot_mae = 0.0

            # Plot — V11 GitHub style: blue solid Actual, red dashed Predicted
            ax.plot(
                df_station.index, df_station[target_col],
                color="#1565C0", linewidth=1.8, label="Actual (Future)",
            )
            ax.plot(
                df_station.index, df_station["prediction"],
                color="#D32F2F", linewidth=1.5, linestyle="--", label="Predicted",
            )

            ax.set_title(
                f"{h}-Day Forecast (MAE={subplot_mae:.1f})",
                fontsize=12, fontweight="bold",
            )
            ax.set_ylabel("PM2.5 (µg/m³)", fontsize=10)
            ax.set_xlabel("Date", fontsize=10)
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.tick_params(axis="x", rotation=30, labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = output_dir / f"forecast_grid_{cc}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    📊 {cc} 2×2 forecast grid → {out_path}")

# ============================================================================
# Terminal Output — Markdown Table
# ============================================================================
def print_results_table(results: list):
    """Print a clean Markdown results table to stdout."""
    valid = [r for r in results if r is not None]
    if not valid:
        print("\n⚠️  No valid results to display.")
        return

    print("\n" + "═" * 98)
    print("  V12 PURE EVALUATION — Phase-Shift Aligned · Nuclear Drop · Zero Imputation")
    print("═" * 98)
    print(f"  Holdout: {HOLDOUT_START} → latest")
    print(f"  Models:  {MODELS_DIR}")
    print("─" * 98)

    header = (
        f"| {'Country':^9} | {'Horizon':^9} | {'Evaluable Samples':^19} "
        f"| {'MAE':^9} | {'NMAE':^8} | {'MASE':^12} | {'Accuracy':^10} |"
    )
    sep = (
        f"|{'-' * 11}|{'-' * 11}|{'-' * 21}"
        f"|{'-' * 11}|{'-' * 10}|{'-' * 14}|{'-' * 12}|"
    )
    print(header)
    print(sep)

    for r in valid:
        # Flag low-sample results
        n = r["samples"]
        n_str = f"{n:>10,}"
        if n < 200:
            n_str += " 🔴"
        else:
            n_str += "   "

        # Flag MASE quality
        mase = r["mase"]
        if np.isnan(mase):
            mase_str = "     N/A   "
        elif mase < 1.0:
            mase_str = f"  {mase:.4f} ✅"
        else:
            mase_str = f"  {mase:.4f} ⚠️"

        acc = r["accuracy"]
        acc_str = f"{acc:>7.1f}%  "

        print(
            f"| {r['country']:^9} | {r['horizon_label']:^9} | {n_str:>19} "
            f"| {r['mae']:>9.2f} | {r['nmae']:>8.4f} | {mase_str:>12} | {acc_str:>10} |"
        )

    print("─" * 98)
    print("  ✅ = MASE < 1.0 (beats persistence naive)   ⚠️ = MASE ≥ 1.0")
    print("  🔴 = < 200 samples (statistically unreliable)")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    print()
    print("╔" + "═" * 62 + "╗")
    print("║   🔬 V12 PURE EVALUATION ENGINE                             ║")
    print("║   Phase-Shift Aligned · Nuclear Drop · Zero Imputation      ║")
    print("╚" + "═" * 62 + "╝")
    print()

    # ── Setup ──────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data (zero imputation) ────────────────────────────────
    df = load_parquet()

    # ── Evaluate all 16 models ─────────────────────────────────────
    all_results = []

    for cc in COUNTRIES:
        df_country = df[df["country_code"] == cc].copy()
        n_stations = df_country["station_id"].nunique()
        holdout_rows = len(df_country[df_country["date"] >= HOLDOUT_START])

        print(f"\n{'─' * 50}")
        print(
            f"🌍 {cc}  ·  {len(df_country):,} total rows  ·  "
            f"{n_stations} stations  ·  {holdout_rows:,} holdout rows"
        )

        for h in HORIZONS:
            model = load_model(cc, h)
            if model is None:
                print(f"    ❌ h={h:>2}d — model not found, skipping")
                all_results.append(None)
                continue

            result = evaluate_single(df_country, cc, h, model)

            if result is None:
                print(f"    ❌ h={h:>2}d — insufficient holdout data")
                all_results.append(None)
            else:
                mase = result["mase"]
                icon = "✅" if (not np.isnan(mase) and mase < 1.0) else "⚠️"
                n_warn = " 🔴" if result["samples"] < 200 else ""
                print(
                    f"    ✓  h={h:>2}d  ·  "
                    f"Samples: {result['samples']:>7,}{n_warn}  ·  "
                    f"MAE: {result['mae']:>8.2f}  ·  "
                    f"MASE: {mase:.4f} {icon}  ·  "
                    f"Acc: {result['accuracy']:.1f}%"
                )
                all_results.append(result)

    # ── Print results table ────────────────────────────────────────
    print_results_table(all_results)

    # ── Generate plots ─────────────────────────────────────────────
    print("📈 Generating plots...")
    plot_overlay(all_results, PLOTS_DIR / "overlay_graph.png")
    plot_error_decay(all_results, PLOTS_DIR / "error_decay.png")
    plot_country_grids(all_results, PLOTS_DIR)

    print("\n✅ V12 Pure Evaluation complete.")
    print(f"   Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
