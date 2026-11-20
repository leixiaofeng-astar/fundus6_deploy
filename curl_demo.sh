#!/usr/bin/env bash
# Upload image -> get image_id -> request prediction. Run from fundus6_deploy.
cd "$(dirname "$0")"
API_BASE=${API_BASE:-http://127.0.0.1:8056}
TOKEN=${TOKEN:-$(cat .api_key)}

# Step 1: upload, extract image_id
IMAGE_ID=$(curl --fail-with-body -sS "$API_BASE/api/v1/images" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@samples/FOJ-1000_L.jpg" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["image_id"])')
echo "image_id=$IMAGE_ID"

# Step 2: predict with image_id
curl --fail-with-body -sS "$API_BASE/api/v1/predictions" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"image_id\":\"$IMAGE_ID\"}" | python3 -m json.tool
