#!/usr/bin/env python3
"""
IndiaAQ Daily Auto-Collector (launchd-safe)
===========================================
Replaces auto_commit.sh to avoid /bin/bash Full Disk Access issues.
Runs data collection + git commit/push entirely in Python.
"""

import subprocess
import os
import sys
from datetime import datetime

PROJECT_DIR = "/Users/divyanshailani/Desktop/pow-eda-pipeline"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "auto_commit.log")

# Ensure log dir exists
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    """Append to log file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line)

def run_cmd(cmd, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd or PROJECT_DIR,
        env={**os.environ, 
             "PATH": "/usr/local/bin:/usr/bin:/bin:/Library/Frameworks/Python.framework/Versions/3.14/bin",
             "OPENAQ_API_KEY": "67e85101744e1e2da4188d0eed8ce8d79fa76fb622c7267038bca4a01860076a",
             "HOME": os.path.expanduser("~")}
    )
    if result.stdout.strip():
        log(f"  stdout: {result.stdout.strip()[:500]}")
    if result.stderr.strip():
        log(f"  stderr: {result.stderr.strip()[:500]}")
    return result.returncode

def main():
    log("=" * 50)
    log("AUTO-COLLECTOR STARTED")
    log("=" * 50)

    # Phase 1: Run daily collector
    python = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
    script = os.path.join(PROJECT_DIR, "scripts", "run_daily_collector.py")
    
    log("Phase 1: Running daily collector...")
    rc = run_cmd(f"{python} {script} --days 7")
    if rc != 0:
        log(f"WARNING: Collector exited with code {rc}")
    else:
        log("Phase 1: Collection complete.")

    # Phase 2: Git commit and push
    log("Phase 2: Git commit & push...")
    collection_log = os.path.join(PROJECT_DIR, "logs", "collection_log.json")
    
    if os.path.exists(collection_log):
        run_cmd("git add logs/collection_log.json")
        
        # Check if there are staged changes
        rc = run_cmd("git diff --cached --quiet")
        if rc != 0:  # rc != 0 means there ARE changes
            date_str = datetime.now().strftime("%Y-%m-%d")
            run_cmd(f'git commit -m "data: daily collection {date_str}"')
            run_cmd("git push origin main")
            log("Pushed to GitHub.")
        else:
            log("No changes to commit.")
    else:
        log("No collection_log.json found, skipping git.")

    log("AUTO-COLLECTOR DONE")
    log("")

if __name__ == "__main__":
    main()
