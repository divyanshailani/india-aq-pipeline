"""
Global AQ Intelligence — Admin Dashboard
==========================================
Local admin UI for managing the prediction pipeline.

Features:
  - Run Pipeline (one-click: fetch → validate → predict)
  - View live accuracy metrics
  - Deploy to Vercel (commit to GitHub)
  - View run history

Usage:
    python scripts/admin_dashboard.py
    # Opens at http://localhost:8050
"""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, date

import psycopg2

# Try importing FastAPI
try:
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Installing FastAPI + uvicorn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "-q"])
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

DB_CONFIG = {
    "dbname": "indiaaq",
    "user": "postgres",
    "password": "8765",
    "host": "localhost",
    "port": 5432,
}

SITE_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "site_data")
SITE_REPO = "/Users/divyanshailani/Desktop/global-aq-intelligence"

app = FastAPI(title="Global AQ Admin")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Pipeline state ────
pipeline_status = {"running": False, "last_log": "", "last_result": None}


def get_db():
    return psycopg2.connect(**DB_CONFIG)


@app.get("/", response_class=HTMLResponse)
async def admin_page():
    return ADMIN_HTML


@app.get("/api/status")
async def get_status():
    conn = get_db()
    cur = conn.cursor()

    # Last run
    cur.execute("""
        SELECT run_id, run_date, predictions_made, validations_done, 
               live_mae, live_r2, status
        FROM pipeline_runs ORDER BY run_date DESC LIMIT 5
    """)
    runs = []
    for r in cur.fetchall():
        runs.append({
            "run_id": str(r[0])[:8],
            "date": str(r[1]),
            "predictions": r[2],
            "validations": r[3],
            "live_mae": r[4],
            "live_r2": r[5],
            "status": r[6],
        })

    # Data stats
    cur.execute("""
        SELECT s.country_code, COUNT(*), MAX(r.datetime_utc)::date
        FROM raw_measurements r JOIN stations s ON r.station_id = s.id
        GROUP BY s.country_code ORDER BY s.country_code
    """)
    data_stats = {}
    for r in cur.fetchall():
        data_stats[r[0]] = {
            "rows": r[1],
            "last_date": str(r[2]),
            "gap_days": (date.today() - r[2]).days
        }

    # Prediction stats
    cur.execute("""
        SELECT country_code, COUNT(*), 
               COUNT(actual_value) as validated,
               AVG(ABS(error)) FILTER (WHERE error IS NOT NULL) as mae
        FROM prediction_log
        GROUP BY country_code
    """)
    pred_stats = {}
    for r in cur.fetchall():
        pred_stats[r[0]] = {
            "total": r[1],
            "validated": r[2],
            "live_mae": round(float(r[3]), 2) if r[3] else None,
        }

    conn.close()

    return {
        "pipeline_running": pipeline_status["running"],
        "runs": runs,
        "data_stats": data_stats,
        "prediction_stats": pred_stats,
    }


@app.post("/api/run-pipeline")
async def run_pipeline(background_tasks: BackgroundTasks):
    if pipeline_status["running"]:
        return {"error": "Pipeline already running"}

    pipeline_status["running"] = True
    pipeline_status["last_log"] = "Starting..."

    background_tasks.add_task(_run_pipeline_bg)
    return {"status": "started"}


def _run_pipeline_bg():
    try:
        script = os.path.join(os.path.dirname(__file__), "predict_pipeline.py")
        result = subprocess.run(
            [sys.executable, script, "--skip-fetch"],
            capture_output=True, text=True, timeout=600,
        )
        pipeline_status["last_log"] = result.stdout + result.stderr
        pipeline_status["last_result"] = "success" if result.returncode == 0 else "error"
    except Exception as e:
        pipeline_status["last_log"] = str(e)
        pipeline_status["last_result"] = "error"
    finally:
        pipeline_status["running"] = False


@app.get("/api/pipeline-log")
async def get_pipeline_log():
    return {
        "running": pipeline_status["running"],
        "log": pipeline_status["last_log"],
        "result": pipeline_status["last_result"],
    }


@app.post("/api/deploy")
async def deploy_to_vercel():
    """Copy predictions to site repo and commit."""
    try:
        # Copy data files
        data_dir = os.path.join(SITE_REPO, "public", "data")
        os.makedirs(data_dir, exist_ok=True)

        for f in os.listdir(SITE_DATA):
            if f.endswith(".json"):
                src = os.path.join(SITE_DATA, f)
                dst = os.path.join(data_dir, f)
                with open(src) as sf:
                    data = sf.read()
                with open(dst, "w") as df:
                    df.write(data)

        # Git commit and push
        today = date.today().isoformat()
        result = subprocess.run(
            ["git", "add", "-A"],
            cwd=SITE_REPO, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"Update predictions [{today}]"],
            cwd=SITE_REPO, capture_output=True, text=True,
        )
        push_result = subprocess.run(
            ["git", "push"],
            cwd=SITE_REPO, capture_output=True, text=True,
        )

        return {
            "status": "deployed",
            "commit_msg": f"Update predictions [{today}]",
            "push": push_result.stdout + push_result.stderr,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── Admin HTML ───────────────────────────────────────────
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global AQ Intelligence — Admin</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            transition: transform 0.2s, border-color 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        .card h3 {
            font-size: 0.85rem;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #00b09b, #96c93d);
            color: white;
        }
        
        .btn-success:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(0, 176, 155, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stat:last-child { border-bottom: none; }
        
        .stat-label { color: #888; font-size: 0.85rem; }
        
        .stat-value {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .stat-value.green { color: #00b09b; }
        .stat-value.yellow { color: #ffc107; }
        .stat-value.red { color: #ff4757; }
        .stat-value.blue { color: #667eea; }
        
        .tag {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .tag-green { background: rgba(0, 176, 155, 0.15); color: #00b09b; }
        .tag-yellow { background: rgba(255, 193, 7, 0.15); color: #ffc107; }
        .tag-blue { background: rgba(102, 126, 234, 0.15); color: #667eea; }
        .tag-red { background: rgba(255, 71, 87, 0.15); color: #ff4757; }
        
        .log-box {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #aaa;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            margin-top: 1rem;
        }
        
        .actions {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .run-history {
            width: 100%;
            border-collapse: collapse;
        }
        
        .run-history th {
            text-align: left;
            padding: 0.5rem;
            color: #888;
            font-size: 0.75rem;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .run-history td {
            padding: 0.5rem;
            font-size: 0.85rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }
        
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Global AQ Admin</h1>
        <p class="subtitle">Manage predictions, validate accuracy, deploy to production</p>
        
        <div class="actions">
            <button class="btn btn-primary" id="runBtn" onclick="runPipeline()">
                ⚡ Run Pipeline
            </button>
            <button class="btn btn-success" id="deployBtn" onclick="deploy()">
                🚀 Deploy to Vercel
            </button>
        </div>
        
        <div class="grid">
            <!-- Data Health -->
            <div class="card">
                <h3>📊 Data Health</h3>
                <div id="dataHealth">Loading...</div>
            </div>
            
            <!-- Prediction Stats -->
            <div class="card">
                <h3>🎯 Prediction Stats</h3>
                <div id="predStats">Loading...</div>
            </div>
        </div>
        
        <!-- Pipeline Log -->
        <div class="card" id="logCard" style="display:none;">
            <h3>📋 Pipeline Log</h3>
            <div class="log-box" id="logBox"></div>
        </div>
        
        <!-- Run History -->
        <div class="card" style="margin-top: 1.5rem;">
            <h3>📜 Run History</h3>
            <table class="run-history" id="runHistory">
                <thead>
                    <tr>
                        <th>Run ID</th>
                        <th>Date</th>
                        <th>Predictions</th>
                        <th>Validated</th>
                        <th>Live MAE</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    
    <script>
        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                // Data Health
                let healthHtml = '';
                const countries = {IN: '🇮🇳', US: '🇺🇸', GB: '🇬🇧', AU: '🇦🇺'};
                for (const [cc, info] of Object.entries(data.data_stats || {})) {
                    const gapClass = info.gap_days <= 1 ? 'green' : info.gap_days <= 7 ? 'yellow' : 'red';
                    healthHtml += `
                        <div class="stat">
                            <span class="stat-label">${countries[cc] || cc} ${cc} — ${(info.rows/1e6).toFixed(1)}M rows</span>
                            <span class="stat-value ${gapClass}">${info.gap_days}d ago</span>
                        </div>`;
                }
                document.getElementById('dataHealth').innerHTML = healthHtml || 'No data';
                
                // Prediction Stats
                let predHtml = '';
                for (const [cc, info] of Object.entries(data.prediction_stats || {})) {
                    predHtml += `
                        <div class="stat">
                            <span class="stat-label">${countries[cc] || cc} ${cc} — ${info.total} preds, ${info.validated} validated</span>
                            <span class="stat-value ${info.live_mae ? 'green' : 'blue'}">${info.live_mae ? info.live_mae + ' µg/m³' : 'pending'}</span>
                        </div>`;
                }
                document.getElementById('predStats').innerHTML = predHtml || 'No predictions yet';
                
                // Run History
                const tbody = document.querySelector('#runHistory tbody');
                tbody.innerHTML = '';
                for (const run of data.runs || []) {
                    const statusTag = run.status === 'completed' 
                        ? '<span class="tag tag-green">✓ Done</span>'
                        : '<span class="tag tag-yellow">Running</span>';
                    tbody.innerHTML += `
                        <tr>
                            <td>${run.run_id}</td>
                            <td>${run.date}</td>
                            <td>${run.predictions || '—'}</td>
                            <td>${run.validations || '—'}</td>
                            <td>${run.live_mae ? run.live_mae.toFixed(2) : '—'}</td>
                            <td>${statusTag}</td>
                        </tr>`;
                }
                
                // Button state
                document.getElementById('runBtn').disabled = data.pipeline_running;
                if (data.pipeline_running) {
                    document.getElementById('runBtn').innerHTML = '<span class="spinner"></span> Running...';
                }
            } catch (e) {
                console.error('Status error:', e);
            }
        }
        
        async function runPipeline() {
            const btn = document.getElementById('runBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Running...';
            document.getElementById('logCard').style.display = 'block';
            document.getElementById('logBox').textContent = 'Starting pipeline...\\n';
            
            await fetch('/api/run-pipeline', {method: 'POST'});
            
            // Poll log
            const pollLog = setInterval(async () => {
                const res = await fetch('/api/pipeline-log');
                const data = await res.json();
                document.getElementById('logBox').textContent = data.log;
                document.getElementById('logBox').scrollTop = 999999;
                
                if (!data.running) {
                    clearInterval(pollLog);
                    btn.disabled = false;
                    btn.innerHTML = '⚡ Run Pipeline';
                    loadStatus();
                }
            }, 2000);
        }
        
        async function deploy() {
            const btn = document.getElementById('deployBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Deploying...';
            
            try {
                const res = await fetch('/api/deploy', {method: 'POST'});
                const data = await res.json();
                
                if (data.status === 'deployed') {
                    alert('✅ Deployed!\\n\\n' + data.commit_msg + '\\n\\n' + data.push);
                } else {
                    alert('❌ Error: ' + (data.error || 'Unknown'));
                }
            } catch (e) {
                alert('❌ Error: ' + e.message);
            }
            
            btn.disabled = false;
            btn.innerHTML = '🚀 Deploy to Vercel';
        }
        
        // Auto-refresh every 10s
        loadStatus();
        setInterval(loadStatus, 10000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Global AQ Intelligence — Admin Dashboard       ║")
    print("║  http://localhost:8050                           ║")
    print("╚══════════════════════════════════════════════════╝")
    uvicorn.run(app, host="0.0.0.0", port=8050)
