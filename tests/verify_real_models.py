"""Opt-in integration: python tests/verify_real_models.py from deployment root."""
import base64
import csv
import json
import os
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from fundus_api.app import create_app
from fundus_api.models import ROOT, ModelService, get_fundus_ai_prediction

service = ModelService('cpu')
samples = sorted((ROOT / 'samples').glob('*.jpg'))
assert samples, 'No sample JPG files'
from unittest.mock import patch
with patch('fundus_api.models.get_iqa_prediction', return_value={
    'poor_quality_probability': 0.95, 'threshold': 0.67, 'quality_score': 0.05, 'gradable': False}):
    poor_result = service.predict(samples[0], {}, 0.67)
    assert poor_result['status'] == 'ungradable'
    assert not poor_result['quality']['gradable']
    assert len(poor_result['diseases']) == 6, 'Ungradable images must retain all disease predictions'

baseline = os.getenv('FUNDUS_BASELINE_CSV')
if baseline:
    with open(baseline) as f:
        for row in csv.DictReader(f):
            result = get_fundus_ai_prediction(row['image'], service.fundus)
            for name, item in result.items():
                assert abs(item['probability'] - float(row['p_'+name])) < 2e-6, (name, row['image'])
with tempfile.TemporaryDirectory() as data:
    with TestClient(create_app(service, data, 'test-secret')) as client:
        headers = {'Authorization': 'Bearer test-secret'}
        ids = []
        for sample in samples:
            response = client.post('/api/v1/image', headers=headers, files={
                'filename': (None, sample.name), 'content': (None, base64.b64encode(sample.read_bytes()).decode()), 'owner': (None, 'integration')})
            assert response.status_code == 200, response.text
            ids.append(response.json()['id'])
        response = client.post('/api/v1/prediction', headers=headers, json={
            'image': {'left': ids[0], 'right': ids[-1]}, 'owner': 'integration', 'patient_id': '',
            'models': ['qc','dr','glaucoma','amd','myopia','cataract','dme']})
        assert response.status_code == 200, response.text
        print(json.dumps(response.json(), indent=2))
        response = client.post('/api/v1/predict', headers=headers, files={'file': (samples[0].name, samples[0].read_bytes())})
        assert response.status_code == 200, response.text
        assert len(response.json()['diseases']) == 6
        saved_output = response.json()
        before = sorted(str(p.relative_to(data)) for p in Path(data).rglob('*'))
        response = client.post('/api/v1/predict/no-save', headers=headers,
            files={'file': (samples[0].name, samples[0].read_bytes())})
        assert response.status_code == 200, response.text
        transient = response.json()
        assert transient['saved'] is False and transient['image_id'] is None and transient['prediction_id'] is None
        assert transient['diseases'] == saved_output['diseases']
        assert transient['quality'] == saved_output['quality']
        assert sorted(str(p.relative_to(data)) for p in Path(data).rglob('*')) == before
        print('Real-model persisted and no-save API integration: passed')
