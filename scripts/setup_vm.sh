#!/bin/bash
# ==============================================================================
# Global AQ Intelligence — VM Setup Script (Ubuntu 24.04 LTS)
# ==============================================================================
# Run this on your Azure VM to configure the FastAPI backend, Nginx, and Systemd.
# Usage: sudo bash setup_vm.sh
# ==============================================================================

echo "🚀 Starting VM Provisioning for Global AQI..."

# 1. Update and install dependencies
echo "📦 Installing system dependencies..."
apt update && apt upgrade -y
apt install -y python3-venv python3-pip postgresql-client nginx git

# 2. Setup Application Directory
APP_DIR="/opt/pow-eda-pipeline"
if [ ! -d "$APP_DIR" ]; then
    echo "📂 Cloning repository..."
    git clone https://github.com/divyanshailani/pow-eda-pipeline.git $APP_DIR
else
    echo "📂 Repository already exists. Pulling latest..."
    cd $APP_DIR && git pull
fi

cd $APP_DIR

# 3. Setup Python Virtual Environment
echo "🐍 Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Create Systemd Service for FastAPI
echo "⚙️ Configuring Systemd Service for FastAPI..."
cat << 'INNER_EOF' > /etc/systemd/system/globalaqi.service
[Unit]
Description=Global AQI FastAPI Service
After=network.target

[Service]
User=root
WorkingDirectory=/opt/pow-eda-pipeline
Environment="PATH=/opt/pow-eda-pipeline/venv/bin"
EnvironmentFile=/opt/pow-eda-pipeline/.env
ExecStart=/opt/pow-eda-pipeline/venv/bin/uvicorn src.app:app --host 127.0.0.1 --port 8000 --workers 2

[Install]
WantedBy=multi-user.target
INNER_EOF

# 5. Configure Nginx Reverse Proxy
echo "🌐 Configuring Nginx Reverse Proxy..."
cat << 'INNER_EOF' > /etc/nginx/sites-available/globalaqi
server {
    listen 80;
    server_name api.globalaqi.live;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
INNER_EOF

ln -sf /etc/nginx/sites-available/globalaqi /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# 6. Enable and Start Services
echo "🔄 Starting Services..."
systemctl daemon-reload
systemctl enable globalaqi
systemctl restart globalaqi
systemctl enable nginx
systemctl restart nginx

echo "✅ VM Setup Complete! FastAPI is running on port 80."
