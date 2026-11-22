#!/usr/bin/env bash
# One request, no image-ID registration and no persistent image/result storage.
# API_BASE=http://server:8056 TOKEN=your-key bash scripts/predict_no_save.sh eye.jpg
set -euo pipefail
: "${API_BASE:=http://127.0.0.1:5050}"
: "${TOKEN:?Set TOKEN to the server credential}"
image_path="${1:-samples/FOJ-1000_L.jpg}"
if [[ ! -s "$image_path" ]]; then
  echo "Image is missing or empty: $image_path" >&2
  exit 1
fi
curl --fail-with-body --silent --show-error --connect-timeout 10 \
  --max-time 120 "${API_BASE%/}/api/v1/predict/no-save" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$image_path"
printf '\n'
