#!/bin/bash
# ==============================================================================
# Global AQ Intelligence — Azure DB Migration Script
# ==============================================================================
# This script securely dumps your local PostgreSQL database and pipes it directly
# into your new Azure PostgreSQL Flexible Server instance.
# ==============================================================================

# Load environment variables from .env securely
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "❌ Error: .env file not found at $ENV_FILE"
    exit 1
fi

# --- Local DB Credentials ---
LOCAL_HOST="localhost"
LOCAL_PORT="5432"
LOCAL_USER="postgres"
LOCAL_DB="indiaaq"
LOCAL_PASSWORD="8765"

# --- Azure DB Credentials ---
AZURE_HOST="globalaqiserver.postgres.database.azure.com"
AZURE_PORT="5432"
AZURE_USER="postgresadmin"
AZURE_PASSWORD="8765@@@###!!!Global"
AZURE_DB="indiaaq"

echo "============================================================"
echo "🚀 Starting Global AQI Database Migration to Azure"
echo "============================================================"

if ! command -v pg_dump &> /dev/null; then
    echo "❌ Error: pg_dump could not be found. Please install PostgreSQL client tools."
    exit 1
fi

echo "📦 Step 1: Creating 'indiaaq' database on Azure (if it doesn't exist)..."
PGPASSWORD="${AZURE_PASSWORD}" psql -h ${AZURE_HOST} -p ${AZURE_PORT} -U ${AZURE_USER} -d postgres -c "CREATE DATABASE ${AZURE_DB};" || true

echo "📦 Step 2: Initiating stream transfer..."
echo "This will dump the local database and pipe it directly to Azure without saving a massive file locally."
echo "Depending on your internet upload speed, transferring 1.6M rows may take 5-15 minutes."

# The --clean flag drops tables on the target before recreating them
# We pass PGPASSWORD inline so they don't overwrite each other in the shell
LOCAL_DUMP_CMD="PGPASSWORD='${LOCAL_PASSWORD}' pg_dump -h ${LOCAL_HOST} -p ${LOCAL_PORT} -U ${LOCAL_USER} -d ${LOCAL_DB} --clean --if-exists --no-owner --no-privileges"
AZURE_RESTORE_CMD="PGPASSWORD='${AZURE_PASSWORD}' psql -h ${AZURE_HOST} -p ${AZURE_PORT} -U ${AZURE_USER} -d ${AZURE_DB}"

# Execute the pipe
eval "${LOCAL_DUMP_CMD} | ${AZURE_RESTORE_CMD}"

if [ ${PIPESTATUS[0]} -eq 0 ] && [ ${PIPESTATUS[1]} -eq 0 ]; then
    echo "✅ Migration Complete!"
    echo "Your local database has been successfully mirrored to Azure."
else
    echo "❌ Migration Failed. Check the error logs above."
fi

unset PGPASSWORD
