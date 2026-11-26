#!/usr/bin/env bash
# Start Z-Image-Turbo API on Ubuntu (H200). Edit the values below.
# Run:  bash cmds/scripts/start_server.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"

export ZIMAGE_MODEL_PATH="${ZIMAGE_MODEL_PATH:-$ROOT/models/Z-Image-Turbo}"
export ZIMAGE_API_KEYS="${ZIMAGE_API_KEYS:-change-me-key1,change-me-key2}"
export ZIMAGE_HOST="${ZIMAGE_HOST:-0.0.0.0}"
export ZIMAGE_PORT="${ZIMAGE_PORT:-8000}"
export ZIMAGE_QUEUE_MAX="${ZIMAGE_QUEUE_MAX:-32}"
export ZIMAGE_MAX_PER_KEY="${ZIMAGE_MAX_PER_KEY:-4}"
export ZIMAGE_JOB_TTL="${ZIMAGE_JOB_TTL:-3600}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$ROOT/zimage_service"
exec python server.py
