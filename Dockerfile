FROM python:3.11-slim

# Install system dependencies required for psycopg2 and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI dashboard with 1 worker to preserve in-memory state
CMD ["uvicorn", "scripts.admin_dashboard:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
