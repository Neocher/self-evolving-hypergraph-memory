# ─── SHM Dockerfile ───
# docker build -t shm:latest .
# docker run -d -p 8000:8000 -v ./data:/app/data --name shm shm:latest

FROM python:3.11-slim

LABEL org.opencontainers.image.title="SHM — Self-evolving Hypergraph Memory"
LABEL org.opencontainers.image.version="5.8.4"

WORKDIR /app

# System deps (Kuzu/FAISS need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY . .

# Create data directory (will be volume-mounted in production)
RUN mkdir -p /app/data/shm_kuzu_db

# Health check: query health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request;d=__import__('json').load(urllib.request.urlopen('http://localhost:8000/health'));exit(0 if d.get('status')=='ok' else 1)"

EXPOSE 8000

# Run migration on startup, then start server
CMD ["python3", "run_server.py"]
