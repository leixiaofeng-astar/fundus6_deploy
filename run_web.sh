#!/usr/bin/env bash
# Start the Fundus web app + REST API on port 8056 (CPU).
cd "$(dirname "$0")"
source .venv/bin/activate
export FUNDUS_DEVICE=cpu
export FUNDUS_API_KEY="$(cat .api_key)"
exec python3 -m uvicorn fundus_api.app:app --host 0.0.0.0 --port 8056 --workers 1 --limit-concurrency 8
