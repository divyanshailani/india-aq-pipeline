"""
Global AQ Intelligence — Shared Configuration
===============================================
Central place for DB credentials & paths.
All scripts import from here instead of hardcoding.

Loads from environment variables first, falls back to defaults
for local development. In production (GitHub Actions), env vars
are set via repository secrets.
"""

import os

# ─── Database ─────────────────────────────────────────────
DB_CONFIG = {
    "dbname": os.environ.get("POSTGRES_DB", "indiaaq"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "8765"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5432)),
}

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
MODEL_DIR = os.path.join(BASE_DIR, "models", "v5")
SITE_DATA_DIR = os.path.join(BASE_DIR, "data", "site_data")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Frontend repo path (for deploy step)
FRONTEND_REPO = os.environ.get(
    "FRONTEND_DATA_PATH",
    "/Users/divyanshailani/Desktop/global-aq-intelligence"
)

# ─── API Keys ─────────────────────────────────────────────
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
AIRNOW_API_KEY = os.environ.get("AIRNOW_API_KEY", "")

# ─── Countries ────────────────────────────────────────────
COUNTRIES = ["IN", "US", "GB", "AU"]
