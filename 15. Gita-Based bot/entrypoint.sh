#!/bin/sh
set -e

# Start server in background bound to requested host:port
# OLLAMA_HOST controls bind address (e.g., 0.0.0.0:11434)
ollama serve &

# Wait for API to come up
sleep 5

# Pull the configured model if not present yet
if [ -n "${MODEL}" ]; then
  echo "Ensuring model ${MODEL} is available..."
  ollama pull "${MODEL}" || true
fi

# Keep foreground
wait
