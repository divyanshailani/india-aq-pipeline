#!/bin/bash
# ============================================================
# IndiaAQ Daily Auto-Collector
# Runs data collection and pushes results to GitHub
# Scheduled via macOS launchd (see com.indiaaq.collector.plist)
# ============================================================

set -e

PROJECT_DIR="/Users/divyanshailani/Desktop/pow-eda-pipeline"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/auto_commit.log"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

mkdir -p "$LOG_DIR"

echo "" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Run: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$PROJECT_DIR"

# Load OpenAQ API key
export OPENAQ_API_KEY="67e85101744e1e2da4188d0eed8ce8d79fa76fb622c7267038bca4a01860076a"

# Run daily collection (last 7 days, all countries)
$PYTHON scripts/run_daily_collector.py --days 7 >> "$LOG_FILE" 2>&1

# Git commit and push if there are changes
if [ -f logs/collection_log.json ]; then
    git add logs/collection_log.json
    git diff --cached --quiet || {
        git commit -m "data: daily collection $(date +%Y-%m-%d)" >> "$LOG_FILE" 2>&1
        git push origin main >> "$LOG_FILE" 2>&1
        echo "Pushed to GitHub" >> "$LOG_FILE"
    }
fi

echo "Done: $(date)" >> "$LOG_FILE"
