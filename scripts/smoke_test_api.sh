#!/usr/bin/env bash
# Run from the deployment root with its Python environment activated.
# TOKEN=your-service-key bash scripts/smoke_test_api.sh
set -euo pipefail
: "${API_BASE:=http://127.0.0.1:5050}"
: "${TOKEN:?Set TOKEN to the configured service credential}"
: "${IMAGE:=samples/FOJ-1000_L.jpg}"
: "${PYTHON:=python}"
API_BASE="${API_BASE%/}"
if [[ ! -s "$IMAGE" ]]; then
  echo "Image is missing or empty: $IMAGE" >&2
  exit 1
fi
curl --fail-with-body --silent --show-error --connect-timeout 10 \
  --max-time 30 "$API_BASE/health"
printf '\n'
response_file=$(mktemp)
trap 'rm -f "$response_file"' EXIT
curl --fail-with-body --silent --show-error --connect-timeout 10 \
  --max-time 120 "$API_BASE/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" -F "file=@$IMAGE" \
  --output "$response_file"
"$PYTHON" - "$response_file" <<'PY'
import json, math, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
print(json.dumps(result, indent=2))
assert result.get('image_id') and result.get('prediction_id'), 'Missing IDs'
quality = result['quality']
assert math.isfinite(quality['poor_quality_probability']), 'Invalid quality output'
assert 0 <= quality['poor_quality_probability'] <= 1, 'Invalid quality probability'
assert result['status'] in ('completed', 'ungradable')
assert quality['gradable'] == (result['status'] == 'completed')
if not quality['gradable']:
    print('WARNING: image is ungradable; all disease outputs are still returned.', file=sys.stderr)
expected = {'dr', 'glaucoma', 'amd', 'myopia', 'cataract', 'dme'}
assert set(result['diseases']) == expected, 'Missing disease heads'
for name, item in result['diseases'].items():
    p, threshold = item['probability'], item['threshold']
    assert math.isfinite(p) and 0 <= p <= 1, name
    assert item['positive'] == (p >= threshold), name
print('PASS: authenticated upload, IQA and all six disease outputs are valid.')
PY
