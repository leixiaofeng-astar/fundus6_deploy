import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from fastapi.testclient import TestClient
from PIL import Image
from fundus_api.app import create_app


class FakeModels:
    def __init__(self, gradable=True):
        self.gradable = gradable
        self.calls = 0

    def predict(self, path, thresholds, iqa_threshold):
        self.calls += 1
        return {'status': 'completed' if self.gradable else 'ungradable',
                'quality': {'poor_quality_probability': 0.1 if self.gradable else 0.9,
                            'threshold': iqa_threshold, 'quality_score': 0.9 if self.gradable else 0.1,
                            'gradable': self.gradable},
                'diseases': {'dr': {'probability': 0.2, 'threshold': thresholds.get('dr', 0.193), 'positive': 0.2 >= thresholds.get('dr', 0.193)}},
                'model_version': 'test', 'iqa_version': 'test'}


def image_bytes():
    out = io.BytesIO()
    Image.new('RGB', (80, 60), '#804433').save(out, format='JPEG')
    return out.getvalue()


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = FakeModels()
        self.app = create_app(self.models, self.temp.name, 'secret')
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.headers = {'Authorization': 'Bearer secret'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self):
        response = self.client.post('/api/v1/images', headers=self.headers, files={'file': ('../eye.jpg', image_bytes())})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['image_id']

    def test_auth(self):
        self.assertEqual(self.client.post('/api/v1/images', files={'file': ('a.jpg', image_bytes())}).status_code, 401)
        self.assertEqual(self.client.post('/api/v1/predict', headers={'X-API-Key': 'secret'}, files={'file': ('a.jpg', image_bytes())}).status_code, 200)

    def test_upload_predict_retrieve_and_restart(self):
        image_id = self.upload()
        response = self.client.post('/api/v1/predictions', headers=self.headers, json={'image_id': image_id, 'thresholds': {'dr': 0.3}})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result['diseases']['dr']['positive'])
        self.assertEqual(self.client.get(f'/api/v1/images/{image_id}/preview', headers=self.headers).status_code, 200)
        with TestClient(create_app(self.models, self.temp.name, 'secret')) as restarted:
            self.assertEqual(restarted.get(f"/api/v1/predictions/{result['prediction_id']}", headers=self.headers).json(), result)

    def test_invalid_image_cleans_files(self):
        response = self.client.post('/api/v1/predict', headers=self.headers, files={'file': ('a.jpg', b'bad')})
        self.assertEqual(response.status_code, 422)
        self.assertEqual([p for p in Path(self.temp.name).iterdir() if p.is_dir()], [])

    def test_validation(self):
        image_id = self.upload()
        for thresholds in ({'unknown': 0.5}, {'dr': 0}, {'dr': 1}):
            self.assertEqual(self.client.post('/api/v1/predictions', headers=self.headers, json={'image_id': image_id, 'thresholds': thresholds}).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/predictions', headers=self.headers, json={'image_id': str(uuid4())}).status_code, 404)
        self.assertEqual(self.models.calls, 0)

    def test_ungradable(self):
        self.models.gradable = False
        response = self.client.post('/api/v1/predict', headers=self.headers, files={'file': ('a.png', image_bytes())})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('dr', response.json()['diseases'])
        self.assertFalse(response.json()['quality']['gradable'])
        self.assertEqual(response.json()['status'], 'ungradable')

    def test_legacy_guide_flow(self):
        response = self.client.post('/api/v1/image', headers=self.headers, files={
            'filename': (None, 'a.jpg'), 'content': (None, base64.b64encode(image_bytes()).decode()), 'owner': (None, 'partner')})
        self.assertEqual(response.status_code, 200, response.text)
        image_id = response.json()['id']
        self.assertEqual(len(image_id), 32)
        response = self.client.post('/api/v1/prediction?owner=partner', headers=self.headers, json={
            'image': {'left': image_id, 'right': image_id}, 'owner': 'partner', 'patient_id': 'external-123', 'models': ['dr', 'qc']})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        results = json.loads(data['results'])
        self.assertIn('DR_probability_l', results)
        self.assertIn('DR_probability_r', results)
        self.assertEqual(data['patient_id'], 'external-123')
        self.assertEqual(self.client.get('/api/v1/prediction/'+data['id'], headers=self.headers).json(), data)
        self.assertGreater(results['QC_probability_l'], 0.5)
        self.assertGreater(results['DR_probability_l'], 0.5)

    def test_legacy_ungradable_keeps_predictions(self):
        self.models.gradable = False
        image_id = self.upload()
        response = self.client.post('/api/v1/prediction', headers=self.headers, json={
            'image': {'left': image_id}, 'owner': 'partner', 'models': ['dr', 'qc']})
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        self.assertFalse(output['eyes']['left']['quality']['gradable'])
        self.assertIn('dr', output['eyes']['left']['diseases'])
        self.assertIsInstance(output['results_json']['DR_probability_l'], float)
        self.assertTrue(output['results_json']['ungradable_l'])
        self.assertEqual(output['results_json']['err_code'], 0)

    def test_invalid_base64_and_models(self):
        response = self.client.post('/api/v1/image', headers=self.headers, files={
            'filename': (None, 'a.jpg'), 'content': (None, '%%%'), 'owner': (None, 'partner')})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post('/api/v1/prediction', headers=self.headers, json={
            'image': {'right': uuid4().hex}, 'owner': 'partner', 'models': ['stroke']}).status_code, 422)

    def test_legacy_large_text_field(self):
        # Regression: Starlette defaults reject multipart text fields above 1 MiB.
        data = image_bytes() + b'\0' * (1024 * 1024)
        response = self.client.post('/api/v1/image', headers=self.headers, files={
            'filename': (None, 'a.jpg'), 'content': (None, base64.b64encode(data).decode()), 'owner': (None, 'partner')})
        self.assertEqual(response.status_code, 200, response.text)

    def test_dicom_upload(self):
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
        ds.Rows = 10; ds.Columns = 10; ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = 'RGB'; ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
        ds.PixelData = np.full((10,10,3), 100, dtype=np.uint8).tobytes()
        ds.is_little_endian = True; ds.is_implicit_VR = False
        out = io.BytesIO(); ds.save_as(out, write_like_original=False)
        response = self.client.post('/api/v1/predict', headers=self.headers, files={'file': ('a.dcm', out.getvalue())})
        self.assertEqual(response.status_code, 200, response.text)

    def test_size_limits(self):
        response = self.client.post('/api/v1/images', headers=self.headers,
            files={'file': ('large.jpg', b'x' * (20*1024*1024 + 1))})
        self.assertEqual(response.status_code, 413)
        response = self.client.post('/api/v1/predict', headers=self.headers,
            content=iter([b'x' * (1024*1024)] * 30))
        self.assertEqual(response.status_code, 413)

    def test_inference_error_is_not_prediction(self):
        from unittest.mock import patch
        with patch.object(self.models, 'predict', side_effect=RuntimeError('failure')):
            response = self.client.post('/api/v1/predict', headers=self.headers,
                files={'file': ('a.jpg', image_bytes())})
        self.assertEqual(response.status_code, 500)
        with self.app.state.store.db() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM predictions').fetchone()[0], 0)

    def test_named_upload_and_duplicate_filenames(self):
        ids = []
        for _ in range(2):
            response = self.client.post('/api/v1/images', headers=self.headers,
                files={'file': ('2026-08-04-14-01-20-722_OS.jpg', image_bytes())})
            self.assertEqual(response.status_code, 201, response.text)
            info = response.json(); ids.append(info['image_id'])
            from uuid import UUID
            self.assertEqual(UUID(info['image_id']).version, 4)
            self.assertEqual(info['stored_original_filename'], '2026-08-04-14-01-20-722_OS__original.jpg')
            self.assertEqual(info['stored_image_filename'], '2026-08-04-14-01-20-722_OS__image.png')
            directory = Path(self.temp.name)/info['image_id']
            self.assertEqual(set(p.name for p in directory.iterdir()),
                {info['stored_original_filename'], info['stored_image_filename']})
            self.assertEqual(self.client.get('/api/v1/images/'+info['image_id']+'/preview', headers=self.headers).status_code, 200)
        self.assertNotEqual(*ids)

    def test_filename_safety_and_old_storage_compatibility(self):
        from fundus_api.app import upload_names
        for name in ['../../escape.jpg', r'C:\folder\eye.jpg', '../', '图像'*150+'.png']:
            _, original, normalized = upload_names(name)
            for saved in [original, normalized]:
                self.assertNotIn('/', saved); self.assertNotIn('\\', saved)
                self.assertLess(len(saved.encode('utf-8')), 255)
        old_id = str(uuid4()); directory = Path(self.temp.name)/old_id
        directory.mkdir(); (directory/'image.png').write_bytes(image_bytes())
        with self.app.state.store.db() as db:
            db.execute('INSERT INTO images VALUES (?,?)', (old_id,json.dumps({'image_id':old_id,'filename':'old.jpg'})))
        self.assertEqual(self.app.state.store.path(old_id), directory/'image.png')
        self.assertEqual(self.client.get('/api/v1/images/'+old_id+'/preview',headers=self.headers).status_code,200)

    def test_no_save_cleans_files_and_does_not_persist(self):
        from unittest.mock import patch
        seen = []
        with patch.object(self.app.state.models, 'predict', wraps=self.models.predict) as model:
            # Capture the transient path while keeping normal fake model output.
            original = model._mock_wraps
            model.side_effect = lambda path, ts, iq: (seen.append(Path(path)), original(path,ts,iq))[1]
            response=self.client.post('/api/v1/predict/no-save',headers=self.headers,files={'file':('eye.jpg',image_bytes())})
        self.assertEqual(response.status_code,200,response.text)
        output=response.json(); self.assertFalse(output['saved'])
        self.assertIsNone(output['image_id']); self.assertIsNone(output['prediction_id'])
        self.assertIn('dr',output['diseases'])
        self.assertTrue(seen); self.assertFalse(seen[0].parent.exists())
        with self.app.state.store.db() as db:
            for table in ['images','predictions','submissions']:
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertFalse(any(p.is_dir() for p in Path(self.temp.name).iterdir()))

    def test_no_save_failure_cleanup_and_auth(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory as RealTemporaryDirectory
        directories=[]
        def temporary(*args,**kwargs):
            manager=RealTemporaryDirectory(*args,**kwargs);directories.append(Path(manager.name));return manager
        for kind in ['bad-image','model-failure','too-large']:
            payload=b'bad' if kind=='bad-image' else b'x'*(20*1024*1024+1) if kind=='too-large' else image_bytes()
            with patch('fundus_api.app.TemporaryDirectory',side_effect=temporary), patch.object(self.models,'predict',side_effect=RuntimeError('failure')):
                response=self.client.post('/api/v1/predict/no-save',headers=self.headers,files={'file':('eye.jpg',payload)})
            self.assertEqual(response.status_code, {'bad-image':422,'model-failure':500,'too-large':413}[kind])
        self.assertTrue(directories);self.assertTrue(all(not d.exists() for d in directories))
        response=self.client.post('/api/v1/predict/no-save',files={'file':('eye.jpg',image_bytes())})
        self.assertEqual(response.status_code,401)

    def test_web_and_schema(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertIn('/api/v1/prediction', self.client.get('/openapi.json').json()['paths'])


if __name__ == '__main__':
    unittest.main()
