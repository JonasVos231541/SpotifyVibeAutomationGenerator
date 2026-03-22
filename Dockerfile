# ── Build stage: install deps ────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py requirements.txt ./
COPY vibe_splitter/ vibe_splitter/

# Create data directory (will be mounted as persistent volume)
RUN mkdir -p /data

# Environment: point all file paths to persistent volume
ENV VS_DB_FILE=/data/vibe_splitter.db \
    VS_STATE_FILE=/data/splitter_state.json \
    VS_MODEL_FILE=/data/splitter_model.npz \
    VS_MODEL_META_FILE=/data/splitter_model_meta.json \
    VS_TFIDF_MODEL_FILE=/data/tfidf_pipeline.pkl \
    VS_CACHE_FILE=/data/track_cache.json \
    VS_LOG_FILE=/data/vibe_splitter.log \
    VS_LOG_FORMAT=json \
    SPOTIPY_CACHE_PATH=/data/.spotify_cache \
    PYTHONUNBUFFERED=1

EXPOSE 10000

# Use Python directly as entrypoint — avoids Windows CRLF issues with shell scripts
CMD ["python", "-c", "import os, secrets; os.makedirs('/data', exist_ok=True); os.environ.setdefault('FLASK_SECRET_KEY', open('/data/.flask_secret').read().strip() if os.path.exists('/data/.flask_secret') else (lambda k: (open('/data/.flask_secret','w').write(k), k)[1])(secrets.token_hex(32))); port = int(os.environ.get('PORT', 10000)); from waitress import serve; from app import app; print(f'Vibe Splitter starting on :{port}', flush=True); serve(app, host='0.0.0.0', port=port, threads=4, channel_timeout=600)"]
