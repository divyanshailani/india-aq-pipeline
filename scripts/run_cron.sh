#!/bin/bash
set -e

# Log all output
exec > >(tee -a /opt/pow-eda-pipeline/logs/cron.log) 2>&1

echo "==========================================="
echo "Started daily prediction at $(date)"

cd /opt/pow-eda-pipeline
git pull origin main

source venv/bin/activate
echo "Exporting Parquet..."
python3 scripts/export_azure_to_parquet.py

echo "Running ONNX Inference..."
python3 scripts/predict_v12_onnx.py

# Publish to Next.js repo via SSH Deploy Key
if [ ! -d "../global-aq-intelligence" ]; then
    echo "Cloning frontend repo..."
    cd ..
    git clone git@github.com:divyanshailani/global-aq-intelligence.git
    cd pow-eda-pipeline
fi

echo "Copying JSON to frontend repo..."
cp site_data/*.json ../global-aq-intelligence/public/data/
cp site_data/model_meta.json ../global-aq-intelligence/public/data/

cd ../global-aq-intelligence
git config user.name "Global AQI VM Bot"
git config user.email "bot@globalaqi.com"
git add public/data/
git commit -m "auto: Daily V12 ONNX Predictions Update 🚀" || echo "No changes to commit"
git push origin main

echo "Finished daily prediction at $(date)"
