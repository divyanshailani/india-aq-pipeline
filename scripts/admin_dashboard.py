"""
Global AQ Intelligence — Admin Control Panel
==============================================
Local admin UI for the full data pipeline.

4-Step Workflow:
  STEP 1: Fetch Latest Data (OpenAQ + NASA POWER)
  STEP 2: Run 30-Day Predictions
  STEP 3: Push to GitHub → Vercel auto-deploys
  STEP 4: Retrain Models (weekly, with R² guard)

Usage:
    python scripts/admin_dashboard.py
    # Opens at http://localhost:8050
"""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, date, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras

try:
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    import uvicorn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "fastapi", "uvicorn[standard]", "-q"])
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    import uvicorn

# ─── Config ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DB_CONFIG, SCRIPTS_DIR, SITE_DATA_DIR, FRONTEND_REPO, MODEL_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DATA = SITE_DATA_DIR
SITE_REPO = FRONTEND_REPO

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")  # optional auth token

app = FastAPI(title="Global AQ Admin")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── Live pipeline state ─────────────────────────────────────
pipeline_state = {
    "running": False,
    "step": None,        # "fetch" | "predict" | "deploy" | "retrain"
    "logs": [],
    "result": None,      # "success" | "error"
    "started_at": None,
}


def get_db():
    return psycopg2.connect(**DB_CONFIG)


def ensure_tracking_schema(conn):
    """Allow the admin panel to read newer metric columns on older local DBs."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE pipeline_runs
                ADD COLUMN IF NOT EXISTS backtest_mae DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS backtest_r2 DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS metric_source TEXT,
                ADD COLUMN IF NOT EXISTS metric_sample_count INTEGER
        """)
    conn.commit()


def add_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    pipeline_state["logs"].append(f"[{ts}] {msg}")


# ═══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def admin_page():
    return ADMIN_HTML


@app.get("/api/status")
async def get_status():
    """Return full system status: data gaps, model info, run history."""
    try:
        conn = get_db()
        ensure_tracking_schema(conn)
        cur = conn.cursor()

        # ── Data gaps per country ──
        cur.execute("""
            SELECT s.country_code,
                   COUNT(DISTINCT r.station_id) as stations,
                   COUNT(*) as rows,
                   MAX(r.datetime_utc)::date as last_date
            FROM raw_measurements r
            JOIN stations s ON r.station_id = s.id
            WHERE r.parameter = 'pm25'
            GROUP BY s.country_code
            ORDER BY s.country_code
        """)
        data_stats = {}
        for r in cur.fetchall():
            gap = (date.today() - r[3]).days if r[3] else None
            data_stats[r[0]] = {
                "stations": r[1],
                "rows": r[2],
                "last_date": str(r[3]) if r[3] else None,
                "gap_days": gap,
            }

        # ── Recent pipeline runs ──
        cur.execute("""
            SELECT run_id, run_date, predictions_made, validations_done,
                   live_mae, live_r2, backtest_mae, backtest_r2,
                   metric_source, metric_sample_count, status
            FROM pipeline_runs
            ORDER BY run_date DESC LIMIT 10
        """)
        runs = []
        for r in cur.fetchall():
            metric_source = r[8] or ("live" if r[4] is not None else None)
            metric_mae = r[4] if metric_source == "live" else r[6]
            metric_r2 = r[5] if metric_source == "live" else r[7]
            runs.append({
                "run_id": str(r[0])[:8],
                "date": str(r[1]),
                "predictions": r[2],
                "validations": r[3],
                "live_mae": round(float(r[4]), 2) if r[4] else None,
                "live_r2": round(float(r[5]), 4) if r[5] else None,
                "backtest_mae": round(float(r[6]), 2) if r[6] else None,
                "backtest_r2": round(float(r[7]), 4) if r[7] else None,
                "metric_source": metric_source,
                "metric_sample_count": r[9],
                "metric_mae": round(float(metric_mae), 2) if metric_mae else None,
                "metric_r2": round(float(metric_r2), 4) if metric_r2 else None,
                "status": r[10],
            })

        # ── Model metadata from JSON ──
        models = {}
        v6_model_dir = os.path.join(BASE_DIR, "models", "v6")
        meta_file = os.path.join(v6_model_dir, "all_models_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                models = json.load(f)

        # ── Last retrain date ──
        last_retrain = None
        for cc in ["IN", "US", "GB", "AU"]:
            pkl = os.path.join(v6_model_dir, f"{cc}_pm25_h1_gbr.pkl")
            if os.path.exists(pkl):
                mtime = datetime.fromtimestamp(os.path.getmtime(pkl))
                if last_retrain is None or mtime > last_retrain:
                    last_retrain = mtime

        # ── Site data freshness ──
        site_meta_path = os.path.join(SITE_DATA, "model_meta.json")
        site_generated = None
        if os.path.exists(site_meta_path):
            with open(site_meta_path) as f:
                sm = json.load(f)
                site_generated = sm.get("generated_at")

        conn.close()

        return {
            "data_stats": data_stats,
            "runs": runs,
            "models": models,
            "last_retrain": last_retrain.isoformat() if last_retrain else None,
            "days_since_retrain": (datetime.now() - last_retrain).days if last_retrain else None,
            "site_generated": site_generated,
            "pipeline": {
                "running": pipeline_state["running"],
                "step": pipeline_state["step"],
                "result": pipeline_state["result"],
            },
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/admin/run-validation")
async def run_live_validation_endpoint():
    try:
        from live_validation import run_live_validation
        count = run_live_validation()
        return {"status": "success", "live_validation_count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/admin/validation-drift")
async def get_validation_drift():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT target_date, AVG(error_delta) as avg_error
                FROM validation_ledger
                GROUP BY target_date
                ORDER BY target_date ASC
            """)
            rows = cur.fetchall()
            data = [{"date": r[0].isoformat(), "avg_error": float(r[1])} for r in rows]
        conn.close()
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/logs")
async def get_logs():
    """Return live pipeline logs."""
    return {
        "running": pipeline_state["running"],
        "step": pipeline_state["step"],
        "logs": pipeline_state["logs"][-200:],
        "result": pipeline_state["result"],
    }


# ─── STEP 1: Fetch Data ──────────────────────────────────────

@app.post("/api/fetch")
async def fetch_data(background_tasks: BackgroundTasks):
    if pipeline_state["running"]:
        return {"error": "Pipeline already running"}
    pipeline_state["running"] = True
    pipeline_state["step"] = "fetch"
    pipeline_state["logs"] = []
    pipeline_state["result"] = None
    pipeline_state["started_at"] = datetime.now().isoformat()
    background_tasks.add_task(_run_fetch)
    return {"status": "started", "step": "fetch"}


def _run_fetch():
    try:
        add_log("═══ STEP 1: FETCH LATEST DATA ═══")

        add_log("Running incremental OpenAQ collector...")
        process = subprocess.Popen(
            [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_daily_collector.py"), "--incremental-only"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=BASE_DIR
        )
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line:
                add_log(f"  {line}")
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            add_log(f"  ⚠️ Collector exited with code {return_code}")

        add_log("Running daily ETL pipeline (Cleaning, Features, Weather, AOD)...")
        process2 = subprocess.Popen(
            [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_daily_etl.py")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=BASE_DIR
        )
        for line in iter(process2.stdout.readline, ""):
            line = line.strip()
            if line:
                add_log(f"  {line}")
        process2.stdout.close()
        return_code = process2.wait()
        if return_code != 0:
            add_log(f"  ⚠️ ETL exited with code {return_code}")

        add_log("✅ FETCH COMPLETE")
        pipeline_state["result"] = "success"
    except Exception as e:
        add_log(f"❌ FETCH FAILED: {e}")
        pipeline_state["result"] = "error"
    finally:
        pipeline_state["running"] = False


# ─── STEP 2: Run Predictions ─────────────────────────────────

@app.post("/api/predict")
async def run_predict(background_tasks: BackgroundTasks):
    if pipeline_state["running"]:
        return {"error": "Pipeline already running"}
    pipeline_state["running"] = True
    pipeline_state["step"] = "predict"
    pipeline_state["logs"] = []
    pipeline_state["result"] = None
    pipeline_state["started_at"] = datetime.now().isoformat()
    background_tasks.add_task(_run_predict)
    return {"status": "started", "step": "predict"}


def _run_predict():
    try:
        add_log("═══ STEP 2: RUN 30-DAY PREDICTIONS ═══")
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "predict_pipeline.py"),
             "--skip-fetch"],
            capture_output=True, text=True, timeout=600,
            cwd=BASE_DIR,
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                add_log(line.strip())

        if result.returncode == 0:
            add_log("✅ PREDICTIONS COMPLETE")

            # Copy to frontend repo
            add_log("Copying JSON to frontend repo...")
            data_dir = os.path.join(SITE_REPO, "public", "data")
            os.makedirs(data_dir, exist_ok=True)
            copied = 0
            for f in os.listdir(SITE_DATA):
                if f.endswith(".json"):
                    shutil.copy2(os.path.join(SITE_DATA, f),
                                 os.path.join(data_dir, f))
                    copied += 1
            add_log(f"  Copied {copied} JSON files → {data_dir}")
            pipeline_state["result"] = "success"
        else:
            add_log(f"❌ Pipeline error: {result.stderr[:500]}")
            pipeline_state["result"] = "error"
    except Exception as e:
        add_log(f"❌ PREDICT FAILED: {e}")
        pipeline_state["result"] = "error"
    finally:
        pipeline_state["running"] = False


# ─── STEP 3: Deploy to Vercel ────────────────────────────────

@app.post("/api/deploy")
async def deploy(background_tasks: BackgroundTasks):
    if pipeline_state["running"]:
        return {"error": "Pipeline already running"}
    pipeline_state["running"] = True
    pipeline_state["step"] = "deploy"
    pipeline_state["logs"] = []
    pipeline_state["result"] = None
    pipeline_state["started_at"] = datetime.now().isoformat()
    background_tasks.add_task(_run_deploy)
    return {"status": "started", "step": "deploy"}


def _run_deploy():
    try:
        add_log("═══ STEP 3: DEPLOY TO VERCEL ═══")
        today = date.today().isoformat()

        # Git add
        add_log("git add -A...")
        subprocess.run(["git", "add", "-A"], cwd=SITE_REPO,
                        capture_output=True, text=True)

        # Git commit
        msg = f"data: update predictions [{today}]"
        add_log(f"git commit -m \"{msg}\"...")
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=SITE_REPO, capture_output=True, text=True,
        )
        add_log(result.stdout.strip() if result.stdout.strip() else
                result.stderr.strip() if result.stderr.strip() else
                "Nothing to commit")

        # Git push
        add_log("git push origin main...")
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=SITE_REPO, capture_output=True, text=True, timeout=60,
        )
        output = (result.stdout + result.stderr).strip()
        for line in output.split("\n"):
            if line.strip():
                add_log(f"  {line.strip()}")

        if result.returncode == 0:
            add_log("✅ PUSHED — Vercel will auto-deploy in ~30s")
            pipeline_state["result"] = "success"
        else:
            add_log(f"⚠️ Push may have failed. Check output above.")
            pipeline_state["result"] = "error"
    except Exception as e:
        add_log(f"❌ DEPLOY FAILED: {e}")
        pipeline_state["result"] = "error"
    finally:
        pipeline_state["running"] = False


# ─── STEP 4: Retrain Models ──────────────────────────────────

@app.post("/api/retrain")
async def retrain(background_tasks: BackgroundTasks):
    if pipeline_state["running"]:
        return {"error": "Pipeline already running"}
    pipeline_state["running"] = True
    pipeline_state["step"] = "retrain"
    pipeline_state["logs"] = []
    pipeline_state["result"] = None
    pipeline_state["started_at"] = datetime.now().isoformat()
    background_tasks.add_task(_run_retrain)
    return {"status": "started", "step": "retrain"}


def _run_retrain():
    try:
        add_log("═══ STEP 4: RETRAIN MODELS ═══")
        add_log("Running retrain_pipeline.py (Continuous Retraining Loop with R² guard)...")

        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "retrain_pipeline.py")],
            capture_output=True, text=True, timeout=1200,
            cwd=BASE_DIR,
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                add_log(line.strip())

        if result.returncode == 0:
            add_log("✅ RETRAIN COMPLETE — new models saved to models/v9/")
            pipeline_state["result"] = "success"
        else:
            add_log(f"❌ Train error: {result.stderr[:500]}")
            pipeline_state["result"] = "error"
    except Exception as e:
        add_log(f"❌ RETRAIN FAILED: {e}")
        pipeline_state["result"] = "error"
    finally:
        pipeline_state["running"] = False


# ─── Full Pipeline (all steps) ───────────────────────────────

@app.post("/api/run-all")
async def run_all(background_tasks: BackgroundTasks):
    if pipeline_state["running"]:
        return {"error": "Pipeline already running"}
    pipeline_state["running"] = True
    pipeline_state["step"] = "full"
    pipeline_state["logs"] = []
    pipeline_state["result"] = None
    pipeline_state["started_at"] = datetime.now().isoformat()
    background_tasks.add_task(_run_all)
    return {"status": "started", "step": "full"}


def _run_all():
    try:
        add_log("═══ FULL PIPELINE: FETCH → PREDICT → DEPLOY ═══")
        _run_fetch()
        pipeline_state["running"] = True
        _run_predict()
        pipeline_state["running"] = True
        _run_deploy()
        add_log("═══ FULL PIPELINE COMPLETE ═══")
    except Exception as e:
        add_log(f"❌ FULL PIPELINE FAILED: {e}")
        pipeline_state["result"] = "error"
    finally:
        pipeline_state["running"] = False


# ═══════════════════════════════════════════════════════════════
#  ADMIN HTML
# ═══════════════════════════════════════════════════════════════
ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global AQ Intelligence — Admin</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg: #0c0e14;
            --card: rgba(255,255,255,0.025);
            --card-border: rgba(255,255,255,0.06);
            --text: #e0e4ef;
            --text-muted: #6b7280;
            --teal: #4fb8b0;
            --purple: #8b7eb8;
            --amber: #d4a24c;
            --red: #ef4444;
            --green: #22c55e;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        .container { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem; }

        /* ── Header ── */
        .header { margin-bottom: 2.5rem; }
        .header h1 {
            font-size: 1.75rem; font-weight: 700;
            background: linear-gradient(135deg, var(--teal), var(--purple));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .header-meta {
            display: flex; gap: 2rem; margin-top: 0.5rem;
            font-size: 0.8rem; color: var(--text-muted);
        }
        .header-meta span { display: flex; align-items: center; gap: 0.35rem; }

        /* ── Grid ── */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.25rem; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
        @media (max-width: 900px) {
            .grid-2, .grid-4 { grid-template-columns: 1fr; }
        }

        /* ── Cards ── */
        .card {
            background: var(--card); border: 1px solid var(--card-border);
            border-radius: 14px; padding: 1.25rem;
            backdrop-filter: blur(8px);
        }
        .card-title {
            font-size: 0.7rem; font-weight: 600; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;
        }

        /* ── Step buttons ── */
        .step-card {
            background: var(--card); border: 1px solid var(--card-border);
            border-radius: 14px; padding: 1.25rem; text-align: center;
            transition: all 0.25s;
        }
        .step-card:hover { border-color: rgba(79,184,176,0.3); transform: translateY(-2px); }
        .step-num {
            font-size: 0.65rem; font-weight: 700; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.5rem;
        }
        .step-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.35rem; }
        .step-desc { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem; line-height: 1.5; }

        .btn {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.6rem 1.25rem; border: none; border-radius: 10px;
            font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 600;
            cursor: pointer; transition: all 0.2s;
        }
        .btn-teal { background: rgba(79,184,176,0.15); color: var(--teal); border: 1px solid rgba(79,184,176,0.3); }
        .btn-teal:hover { background: rgba(79,184,176,0.25); box-shadow: 0 4px 20px rgba(79,184,176,0.15); }
        .btn-purple { background: rgba(139,126,184,0.15); color: var(--purple); border: 1px solid rgba(139,126,184,0.3); }
        .btn-purple:hover { background: rgba(139,126,184,0.25); box-shadow: 0 4px 20px rgba(139,126,184,0.15); }
        .btn-amber { background: rgba(212,162,76,0.15); color: var(--amber); border: 1px solid rgba(212,162,76,0.3); }
        .btn-amber:hover { background: rgba(212,162,76,0.25); box-shadow: 0 4px 20px rgba(212,162,76,0.15); }
        .btn-green { background: rgba(34,197,94,0.15); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
        .btn-green:hover { background: rgba(34,197,94,0.25); box-shadow: 0 4px 20px rgba(34,197,94,0.15); }
        .btn-full {
            display: flex; align-items: center; justify-content: center; gap: 0.4rem;
            width: 100%; padding: 0.8rem 1.5rem; margin-top: 1rem;
            border: 1px solid rgba(79,184,176,0.4); border-radius: 12px;
            background: linear-gradient(135deg, rgba(79,184,176,0.12), rgba(139,126,184,0.08));
            color: var(--teal); font-family: 'Inter'; font-size: 0.9rem; font-weight: 600;
            cursor: pointer; transition: all 0.2s;
        }
        .btn-full:hover { background: linear-gradient(135deg, rgba(79,184,176,0.2), rgba(139,126,184,0.15)); box-shadow: 0 6px 30px rgba(79,184,176,0.12); }
        .btn:disabled, .btn-full:disabled { opacity: 0.4; cursor: not-allowed; transform: none !important; box-shadow: none !important; }

        /* ── Country stats ── */
        .country-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .country-row:last-child { border-bottom: none; }
        .country-name { font-size: 0.85rem; font-weight: 500; }
        .country-detail { font-size: 0.72rem; color: var(--text-muted); }
        .gap-badge {
            font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem;
            border-radius: 6px;
        }
        .gap-ok { background: rgba(34,197,94,0.12); color: var(--green); }
        .gap-warn { background: rgba(212,162,76,0.12); color: var(--amber); }
        .gap-bad { background: rgba(239,68,68,0.12); color: var(--red); }

        /* ── Model cards ── */
        .model-metric { display: flex; justify-content: space-between; padding: 0.4rem 0; }
        .model-metric-label { font-size: 0.78rem; color: var(--text-muted); }
        .model-metric-value { font-size: 0.85rem; font-weight: 600; }

        /* ── Log box ── */
        .log-container {
            background: rgba(0,0,0,0.4); border: 1px solid var(--card-border);
            border-radius: 12px; padding: 1rem; margin-top: 1.25rem;
        }
        .log-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0.75rem;
        }
        .log-title { font-size: 0.7rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
        .log-status {
            font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem;
            border-radius: 6px;
        }
        .log-running { background: rgba(79,184,176,0.15); color: var(--teal); animation: pulse 1.5s infinite; }
        .log-success { background: rgba(34,197,94,0.15); color: var(--green); }
        .log-error { background: rgba(239,68,68,0.15); color: var(--red); }
        .log-box {
            font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
            color: #9ca3af; max-height: 350px; overflow-y: auto;
            white-space: pre-wrap; line-height: 1.7;
        }
        .log-box .success { color: var(--green); }
        .log-box .error { color: var(--red); }

        /* ── Run history table ── */
        .table { width: 100%; border-collapse: collapse; }
        .table th {
            text-align: left; padding: 0.5rem 0.75rem;
            font-size: 0.68rem; font-weight: 600; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.06em;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .table td {
            padding: 0.5rem 0.75rem; font-size: 0.8rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }
        .tag {
            display: inline-block; padding: 0.15rem 0.5rem;
            border-radius: 6px; font-size: 0.7rem; font-weight: 600;
        }
        .tag-green { background: rgba(34,197,94,0.12); color: var(--green); }
        .tag-amber { background: rgba(212,162,76,0.12); color: var(--amber); }
        .tag-red { background: rgba(239,68,68,0.12); color: var(--red); }

        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.15); border-top-color: var(--teal); border-radius: 50%; animation: spin 0.7s linear infinite; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🔧 Global AQ Intelligence — Admin</h1>
            <div class="header-meta">
                <span id="lastRun">Last run: loading...</span>
                <span id="siteAge">Site data: loading...</span>
                <span id="retrainInfo">Retrain: loading...</span>
            </div>
        </div>

        <!-- 4 Step Buttons -->
        <div class="grid-4">
            <div class="step-card">
                <div class="step-num">Step 1</div>
                <div class="step-title">📡 Fetch Data</div>
                <div class="step-desc">OpenAQ + NASA POWER<br>weather for all 4 countries</div>
                <button class="btn btn-teal" id="fetchBtn" onclick="runStep('fetch')">▶ Fetch</button>
            </div>
            <div class="step-card">
                <div class="step-num">Step 2</div>
                <div class="step-title">🧠 Predict</div>
                <div class="step-desc">30-day chained forecast<br>validate old predictions</div>
                <button class="btn btn-purple" id="predictBtn" onclick="runStep('predict')">▶ Predict</button>
            </div>
            <div class="step-card">
                <div class="step-num">Step 3</div>
                <div class="step-title">🚀 Deploy</div>
                <div class="step-desc">Push JSON to GitHub<br>Vercel auto-deploys</div>
                <button class="btn btn-green" id="deployBtn" onclick="runStep('deploy')">▶ Deploy</button>
            </div>
            <div class="step-card">
                <div class="step-num">Weekly</div>
                <div class="step-title">🔄 Retrain</div>
                <div class="step-desc">Full model retrain<br>with R² guard</div>
                <button class="btn btn-amber" id="retrainBtn" onclick="runStep('retrain')">▶ Retrain</button>
            </div>
        </div>

        <!-- Full Pipeline Button -->
        <button class="btn-full" id="fullBtn" onclick="runStep('run-all')">
            ⚡ Run Full Pipeline (Fetch → Predict → Deploy)
        </button>

        <!-- Live Validation Observatory -->
        <div class="log-container" style="border: 1px solid rgba(16,185,129,0.3); background: rgba(16,185,129,0.05); padding: 1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <h3 style="font-size:1rem; font-weight:700; color:var(--green); margin:0;">Live Validation Observatory</h3>
            </div>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-top:0; margin-bottom:1rem; line-height:1.5;">
                Force the collision engine to merge today's actuals against predictions and update the ledger.
            </p>
            <button class="btn btn-green" style="width:100%; justify-content:center;" id="btn-validation" onclick="runValidation()">
                Force Run Collision Engine
            </button>
            <div id="validation-msg" style="margin-top:0.75rem; font-size:0.8rem; font-family:'JetBrains Mono'; display:none;"></div>
            
            <!-- Drift Chart -->
            <div style="margin-top: 1.5rem; border-top: 1px solid rgba(16,185,129,0.2); padding-top: 1rem;">
                <h4 style="font-size:0.85rem; font-weight:600; color:var(--text-muted); margin-bottom:0.5rem; margin-top:0;">Model Drift (MAE over time)</h4>
                <div id="drift-status" style="font-size:0.8rem; color:var(--text-muted);">Awaiting first collision data to plot drift.</div>
                <div style="height: 200px; width: 100%;">
                    <canvas id="driftChart" style="display:none;"></canvas>
                </div>
            </div>
        </div>

        <!-- Log Box -->
        <div class="log-container" id="logContainer" style="display:none;">
            <div class="log-header">
                <span class="log-title">📋 Pipeline Log</span>
                <span class="log-status" id="logStatus"></span>
            </div>
            <div class="log-box" id="logBox"></div>
        </div>

        <!-- Data Health + Model Performance -->
        <div class="grid-2" style="margin-top: 1.5rem;">
            <div class="card">
                <div class="card-title">📊 Data Health — PM2.5 Coverage</div>
                <div id="dataHealth">Loading...</div>
            </div>
            <div class="card">
                <div class="card-title">🧠 Model Performance (v6)</div>
                <div id="modelPerf">Loading...</div>
            </div>
        </div>

        <!-- Run History -->
        <div class="card" style="margin-top: 1.25rem;">
            <div class="card-title">📜 Run History</div>
            <table class="table">
                <thead>
                    <tr>
                        <th>Run</th><th>Date</th><th>Predictions</th>
                        <th>Validated</th><th>Metric</th><th>MAE</th><th>R²</th><th>Status</th>
                    </tr>
                </thead>
                <tbody id="runHistory"></tbody>
            </table>
        </div>

        <!-- V9 Live Validation -->
        <div class="card">
            <div class="card-title" style="color: #4fb8b0;">⚡ V9 XGBoost Live Validation (Real-World Accuracy)</div>
            <div class="card-subtitle">
                Tracking the daily live MAE error between V9 XGBoost predictions and actual PM2.5 readings.
            </div>
            <table class="table">
                <thead>
                    <tr>
                        <th>Country</th><th>Samples Validated</th><th>Live MAE (µg/m³)</th><th>Live R²</th>
                    </tr>
                </thead>
                <tbody id="v8Validation">
                    <!-- Populated by JS -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const FLAGS = {IN:'🇮🇳', US:'🇺🇸', GB:'🇬🇧', AU:'🇦🇺'};
        const NAMES = {IN:'India', US:'United States', GB:'United Kingdom', AU:'Australia'};
        let pollInterval = null;

        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const d = await res.json();
                if (d.error) { console.error(d.error); return; }

                // Header meta
                const lastRun = d.runs?.[0];
                document.getElementById('lastRun').textContent =
                    lastRun ? `Last run: ${lastRun.date}` : 'Last run: never';
                document.getElementById('siteAge').textContent =
                    d.site_generated ? `Site data: ${d.site_generated.slice(0,10)}` : 'Site data: —';
                document.getElementById('retrainInfo').textContent =
                    d.days_since_retrain != null
                        ? `Retrain: ${d.days_since_retrain}d ago` + (d.days_since_retrain >= 7 ? ' ⚠️' : ' ✓')
                        : 'Retrain: never';

                // Data Health
                let dh = '';
                for (const cc of ['IN','US','GB','AU']) {
                    const s = d.data_stats?.[cc];
                    if (!s) { dh += `<div class="country-row"><span class="country-name">${FLAGS[cc]} ${cc}</span><span class="gap-badge gap-bad">No data</span></div>`; continue; }
                    const gapClass = s.gap_days <= 2 ? 'gap-ok' : s.gap_days <= 7 ? 'gap-warn' : 'gap-bad';
                    const rows = s.rows > 1e6 ? (s.rows/1e6).toFixed(1)+'M' : s.rows > 1e3 ? (s.rows/1e3).toFixed(0)+'K' : s.rows;
                    dh += `<div class="country-row">
                        <div>
                            <div class="country-name">${FLAGS[cc]} ${NAMES[cc]}</div>
                            <div class="country-detail">${s.stations} stations · ${rows} rows · last: ${s.last_date}</div>
                        </div>
                        <span class="gap-badge ${gapClass}">${s.gap_days}d gap</span>
                    </div>`;
                }
                document.getElementById('dataHealth').innerHTML = dh || 'No data';

                // Model Performance
                let mp = '';
                const models = d.models;
                for (const cc of ['IN','US','GB','AU']) {
                    const m = models?.[cc]?.h1?.metrics || models?.[cc];
                    const mae = m?.test_mae ?? m?.mae ?? '—';
                    const mase = m?.mase ?? '—';
                    
                    let maseColor = 'var(--text-secondary)';
                    if (typeof mase === 'number') {
                        if (mase < 1.0) maseColor = 'var(--green)';
                        else if (mase <= 1.1) maseColor = 'var(--amber)';
                        else maseColor = 'var(--red)';
                    }
                    
                    mp += `<div class="country-row">
                        <div>
                            <div class="country-name">${FLAGS[cc]} ${NAMES[cc]}</div>
                            <div class="country-detail">MAE: ${typeof mae === 'number' ? mae.toFixed(2) : mae} µg/m³</div>
                        </div>
                        <span style="font-size:0.9rem;font-weight:700;color:${maseColor}">MASE ${typeof mase === 'number' ? mase.toFixed(2) : mase}</span>
                    </div>`;
                }
                document.getElementById('modelPerf').innerHTML = mp || 'No model data';

                // Run History
                const tbody = document.getElementById('runHistory');
                tbody.innerHTML = '';
                for (const r of (d.runs || [])) {
                    const tagClass = r.status === 'completed' ? 'tag-green' : r.status === 'running' ? 'tag-amber' : 'tag-red';
                    const tagText = r.status === 'completed' ? '✓ Done' : r.status === 'running' ? '⏳ Running' : '✗ Error';
                    const source = r.metric_source ? `${r.metric_source}${r.metric_sample_count ? ` (${r.metric_sample_count})` : ''}` : '—';
                    tbody.innerHTML += `<tr>
                        <td style="font-family:'JetBrains Mono';font-size:0.75rem;">${r.run_id}</td>
                        <td>${r.date}</td>
                        <td>${r.predictions?.toLocaleString() ?? '—'}</td>
                        <td>${r.validations ?? '—'}</td>
                        <td>${source}</td>
                        <td>${r.metric_mae ?? '—'}</td>
                        <td>${r.metric_r2 ?? '—'}</td>
                        <td><span class="tag ${tagClass}">${tagText}</span></td>
                    </tr>`;
                }

                // Button states
                const running = d.pipeline?.running;
                const btns = ['fetchBtn','predictBtn','deployBtn','retrainBtn','fullBtn'];
                btns.forEach(id => { document.getElementById(id).disabled = running; });
            } catch(e) { console.error('Status load error:', e); }
        }

        async function runValidation() {
            const btn = document.getElementById('btn-validation');
            const msg = document.getElementById('validation-msg');
            btn.disabled = true;
            btn.innerHTML = '⏳ Executing Time-Travel Merge...';
            msg.style.display = 'none';
            try {
                const res = await fetch('/api/admin/run-validation', {method: 'POST'});
                const data = await res.json();
                msg.style.display = 'block';
                if (res.ok && data.status === 'success') {
                    msg.innerHTML = `<span style="color:var(--green)">✅ Success! Total Validation Rows: ${data.live_validation_count}</span>`;
                    loadDrift(); // Refresh chart on success
                } else {
                    msg.innerHTML = `<span style="color:var(--red)">❌ Error: ${data.message || 'Unknown server error'}</span>`;
                }
            } catch(e) {
                msg.style.display = 'block';
                msg.innerHTML = `<span style="color:var(--red)">❌ Network Error: ${e.message}</span>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = 'Force Run Collision Engine';
            }
        }

        async function runStep(step) {
            const btns = ['fetchBtn','predictBtn','deployBtn','retrainBtn','fullBtn'];
            btns.forEach(id => { document.getElementById(id).disabled = true; });

            document.getElementById('logContainer').style.display = 'block';
            document.getElementById('logBox').textContent = 'Starting...';
            document.getElementById('logStatus').className = 'log-status log-running';
            document.getElementById('logStatus').textContent = '⏳ Running';

            await fetch(`/api/${step}`, {method: 'POST'});

            // Poll logs
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(async () => {
                const res = await fetch('/api/logs');
                const data = await res.json();

                let html = '';
                for (const line of data.logs) {
                    if (line.includes('✅')) html += `<span class="success">${line}</span>\\n`;
                    else if (line.includes('❌')) html += `<span class="error">${line}</span>\\n`;
                    else html += line + '\\n';
                }
                document.getElementById('logBox').innerHTML = html;
                document.getElementById('logBox').scrollTop = 999999;

                if (!data.running) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    const status = document.getElementById('logStatus');
                    if (data.result === 'success') {
                        status.className = 'log-status log-success';
                        status.textContent = '✅ Complete';
                    } else {
                        status.className = 'log-status log-error';
                        status.textContent = '❌ Error';
                    }
                    btns.forEach(id => { document.getElementById(id).disabled = false; });
                    loadStatus();
                }
            }, 1500);
        }

        async function loadDrift() {
            try {
                const res = await fetch('/api/admin/validation-drift');
                const data = await res.json();
                if (data && data.length > 0) {
                    document.getElementById('drift-status').style.display = 'none';
                    document.getElementById('driftChart').style.display = 'block';
                    
                    const ctx = document.getElementById('driftChart').getContext('2d');
                    if (window.driftChartInstance) window.driftChartInstance.destroy();
                    
                    window.driftChartInstance = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.map(d => d.date),
                            datasets: [{
                                label: 'Avg Absolute Error',
                                data: data.map(d => d.avg_error),
                                borderColor: '#ef4444',
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.3,
                                pointRadius: 3
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: {size: 10} } },
                                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: {size: 10} } }
                            }
                        }
                    });
                }
            } catch(e) { console.error('Drift load error:', e); }
        }

        loadStatus();
        loadDrift();
        setInterval(loadStatus, 15000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  Global AQ Intelligence — Admin Control Panel   ║")
    print("  ║  http://localhost:8050                           ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  STEP 1: Fetch Data   (OpenAQ + NASA POWER)     ║")
    print("  ║  STEP 2: Predict      (30-day chained forecast) ║")
    print("  ║  STEP 3: Deploy       (git push → Vercel)       ║")
    print("  ║  STEP 4: Retrain      (weekly, R² guard)        ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8050)
