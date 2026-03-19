#!/bin/sh
set -e

# Ensure data directory exists
mkdir -p /data

# Cache HuggingFace models to persistent volume (downloads ~80MB on first run)
export HF_HOME="/data/.hf_cache"

# Symlink spotify cache into data dir so it persists
export SPOTIPY_CACHE_PATH="/data/.spotify_cache"

# Generate flask secret if not set via env
if [ -z "$FLASK_SECRET_KEY" ]; then
    if [ -f /data/.flask_secret ]; then
        export FLASK_SECRET_KEY=$(cat /data/.flask_secret)
    else
        export FLASK_SECRET_KEY=$(python -c "import secrets; s=secrets.token_hex(32); print(s)")
        echo "$FLASK_SECRET_KEY" > /data/.flask_secret
    fi
fi

# Start the app on port 8080 (Fly.io default)
exec python -c "
from waitress import serve
from app import app
print('Vibe Splitter starting on :8080', flush=True)
serve(app, host='0.0.0.0', port=8080, threads=8, channel_timeout=600)
"
