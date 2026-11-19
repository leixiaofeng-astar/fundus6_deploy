"""Run: uvicorn fundus_api.app:app --host 127.0.0.1 --port 8000"""
from typing import Dict, List, Optional
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4, UUID
import base64
import binascii
import io
import time
import hashlib
import hmac
import json
import logging
import os
import sqlite3

from fastapi import FastAPI, Depends, File, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from .imaging import read_image

LOG = logging.getLogger(__name__)
MAX_UPLOAD = 20 * 1024 * 1024


class BodyLimit:
    """Bound the whole body, including chunked uploads, before multipart parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            size += len(message.get('body', b''))
            if size > (MAX_UPLOAD * 4 // 3) + 2 * 1024 * 1024:
                response = JSONResponse({'detail': 'Request exceeds multipart body limit'}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(message.get('body', b''))
            if not message.get('more_body', False):
                break
        payload = b''.join(chunks)
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': payload, 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image_id: UUID
    thresholds: Dict[str, float] = Field(default_factory=dict)
    iqa_threshold: float = Field(default=0.67, gt=0, lt=1, allow_inf_nan=False)

    @field_validator('thresholds')
    @classmethod
    def validate_thresholds(cls, values):
        import math
        known = {'dr', 'glaucoma', 'amd', 'myopia', 'cataract', 'dme'}
        if values.keys() - known or any(not math.isfinite(v) or not 0 < v < 1 for v in values.values()):
            raise ValueError('Use known disease names and finite thresholds strictly between 0 and 1')
        return values


class LegacyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image: Dict[str, str]
    models: List[str] = Field(default_factory=lambda: ['qc', 'dr', 'glaucoma', 'amd', 'myopia', 'cataract', 'dme'])
    patient_id: str = Field(default='', max_length=256)
    owner: str = Field(min_length=1, max_length=256)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    iqa_threshold: float = Field(default=0.67, gt=0, lt=1, allow_inf_nan=False)

    @field_validator('image')
    @classmethod
    def validate_image(cls, values):
        if not values or values.keys() - {'left', 'right'} or not any(values.values()):
            raise ValueError('Supply left and/or right image IDs')
        for value in values.values():
            if value:
                UUID(value)
        return values

    @field_validator('models')
    @classmethod
    def validate_models(cls, values):
        if not values or set(values) - {'qc', 'dr', 'glaucoma', 'amd', 'myopia', 'cataract', 'dme'}:
            raise ValueError('Unknown or empty model selection')
        return list(dict.fromkeys(values))

    @field_validator('thresholds')
    @classmethod
    def validate_thresholds(cls, values):
        return PredictRequest.validate_thresholds(values)


class ImageInfo(BaseModel):
    image_id: UUID
    filename: str
    width: int
    height: int
    size_bytes: int
    sha256: str
    created_at: str


class DiseaseResult(BaseModel):
    probability: float
    threshold: float
    positive: bool


class QualityResult(BaseModel):
    poor_quality_probability: float
    threshold: float
    quality_score: float
    gradable: bool


class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction_id: UUID
    image_id: UUID
    status: str
    quality: QualityResult
    diseases: Optional[Dict[str, DiseaseResult]]
    model_version: str
    iqa_version: str
    created_at: str


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, image_id TEXT NOT NULL, result TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id));
                CREATE INDEX IF NOT EXISTS predictions_image ON predictions(image_id);
                CREATE TABLE IF NOT EXISTS submissions (id TEXT PRIMARY KEY, result TEXT NOT NULL);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / 'index.sqlite3', timeout=30)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def image(self, image_id):
        image_id = str(UUID(str(image_id)))
        with self.db() as db:
            row = db.execute('SELECT metadata FROM images WHERE id=?', (image_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Image not found')
        return json.loads(row[0])

    def path(self, image_id):
        self.image(image_id)
        return self.root / str(UUID(str(image_id))) / 'image.png'

    def upload(self, upload, extra=None):
        image_id = str(uuid4())
        directory = self.root / image_id
        directory.mkdir(mode=0o700)
        try:
            size, digest = 0, hashlib.sha256()
            with (directory / 'original').open('wb') as out:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'Image exceeds 20 MiB limit')
                    digest.update(chunk)
                    out.write(chunk)
            if not size:
                raise HTTPException(422, 'Empty image')
            try:
                im = read_image(directory / 'original')
                im.save(directory / 'image.png')  # no source DICOM metadata in inference image
            except Exception as e:
                raise HTTPException(422, 'Invalid or unsupported image/DICOM') from e
            metadata = {'image_id': image_id, 'filename': (upload.filename or 'image').replace('\\', '/').split('/')[-1],
                        'width': im.width, 'height': im.height, 'size_bytes': size,
                        'sha256': digest.hexdigest(), 'created_at': now()}
            if extra:
                metadata.update(extra)
            with self.db() as db:
                db.execute('INSERT INTO images VALUES (?,?)', (image_id, json.dumps(metadata)))
            return metadata
        except BaseException:
            for file in directory.iterdir():
                file.unlink()
            directory.rmdir()
            raise
        finally:
            upload.file.close()

    def save_result(self, result):
        with self.db() as db:
            db.execute('INSERT INTO predictions VALUES (?,?,?)',
                       (result['prediction_id'], result['image_id'], json.dumps(result, allow_nan=False)))

    def result(self, prediction_id):
        with self.db() as db:
            row = db.execute('SELECT result FROM predictions WHERE id=?', (str(prediction_id),)).fetchone()
        if not row:
            raise HTTPException(404, 'Prediction not found')
        return json.loads(row[0])


def now():
    return datetime.now(timezone.utc).isoformat()


def create_app(model_service=None, data_dir=None, api_key=None):
    key = api_key if api_key is not None else os.getenv('FUNDUS_API_KEY', '')
    bearer = os.getenv('FUNDUS_BEARER_TOKEN', key)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(data_dir or os.getenv('FUNDUS_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data/api')))
        if model_service is None:
            from .models import ModelService
            app.state.models = await run_in_threadpool(ModelService)
        else:
            app.state.models = model_service
        yield

    app = FastAPI(title='Fundus AI REST API', version='1.0.0', lifespan=lifespan,
                  description='Six independent binary disease heads with advisory image quality assessment. Research use only.')
    app.add_middleware(BodyLimit)

    api_key_scheme = APIKeyHeader(name='X-API-Key', auto_error=False)
    bearer_scheme = HTTPBearer(auto_error=False)

    def authenticate(request: Request, api_key_header=Security(api_key_scheme), bearer_header=Security(bearer_scheme)):
        if key or bearer:
            supplied = request.headers.get('x-api-key', '')
            authorization = request.headers.get('authorization', '')
            parts = authorization.split(' ', 1)
            token = parts[1] if len(parts) == 2 and parts[0].lower() == 'bearer' else ''
            valid = (bool(key) and hmac.compare_digest(supplied.encode(), key.encode())) or (bool(bearer) and hmac.compare_digest(token.encode(), bearer.encode()))
            if not valid:
                raise HTTPException(401, 'Invalid or missing API credential', headers={'WWW-Authenticate': 'Bearer'})

    auth = [Depends(authenticate)]

    def predict(body):
        path = app.state.store.path(body.image_id)
        try:
            result = app.state.models.predict(path, body.thresholds, body.iqa_threshold)
            result = {**result, 'prediction_id': str(uuid4()), 'image_id': str(body.image_id), 'created_at': now()}
            # Validate before persistence, preventing partial/invalid model output being recorded.
            result = PredictionResult.model_validate(result).model_dump(mode='json')
            app.state.store.save_result(result)
            return result
        except Exception as e:
            LOG.exception('Inference failed for image_id=%s', body.image_id)
            raise HTTPException(500, 'Model inference failed; image remains available for retry') from e

    @app.get('/', include_in_schema=False)
    def web():
        return FileResponse(Path(__file__).parent / 'static/index.html', headers={'Cache-Control': 'no-store'})

    @app.get('/health')
    def health():
        return {'status': 'ready', 'models_loaded': True}

    @app.post('/api/v1/images', status_code=201, response_model=ImageInfo, dependencies=auth)
    def upload_image(file: UploadFile = File(...)):
        return app.state.store.upload(file)

    @app.get('/api/v1/images/{image_id}', response_model=ImageInfo, dependencies=auth)
    def get_image(image_id: UUID):
        return app.state.store.image(image_id)

    @app.get('/api/v1/images/{image_id}/preview', dependencies=auth)
    def preview(image_id: UUID):
        return FileResponse(app.state.store.path(image_id), media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.post('/api/v1/predictions', response_model=PredictionResult, dependencies=auth)
    def prediction(body: PredictRequest):
        return predict(body)

    @app.get('/api/v1/predictions/{prediction_id}', response_model=PredictionResult, dependencies=auth)
    def get_prediction(prediction_id: UUID):
        return app.state.store.result(prediction_id)

    @app.post('/api/v1/predict', response_model=PredictionResult, dependencies=auth)
    def upload_and_predict(file: UploadFile = File(...)):
        image = app.state.store.upload(file)
        return predict(PredictRequest(image_id=image['image_id']))

    @app.post('/api/v1/image', status_code=200, dependencies=auth, openapi_extra={
        'requestBody': {'required': True, 'content': {'multipart/form-data': {'schema': {
            'type': 'object', 'required': ['filename', 'content', 'owner'], 'properties': {
                'filename': {'type': 'string'}, 'content': {'type': 'string', 'description': 'Base64-encoded image bytes'},
                'owner': {'type': 'string'}, 'name': {'type': 'string'}, 'model': {'type': 'string'}}}}}}})
    async def legacy_upload(request: Request):
        # Explicit multipart parser limits: a 20 MiB file expands to ~27 MiB base64.
        async with request.form(max_files=0, max_fields=8, max_part_size=MAX_UPLOAD * 4 // 3 + 4) as form:
            filename, content, owner = (form.get(k) for k in ('filename', 'content', 'owner'))
            if not all(isinstance(v, str) and v for v in (filename, content, owner)):
                raise HTTPException(422, 'filename, content and owner text fields are required')
            if len(filename) > 1024 or len(owner) > 256:
                raise HTTPException(422, 'Metadata field is too long')
            try:
                data = base64.b64decode(''.join(content.split()), validate=True)
            except (ValueError, binascii.Error) as e:
                raise HTTPException(422, 'Invalid Base64 content') from e
            upload = UploadFile(file=io.BytesIO(data), filename=filename)
            info = await run_in_threadpool(app.state.store.upload, upload, {'owner': owner, 'name': str(form.get('name', filename)), 'model': str(form.get('model', ''))})
            # Keep the original guide's 32-character UUID representation.
            return {**info, 'id': UUID(info['image_id']).hex, 'owner': owner,
                    'name': str(form.get('name', filename)), 'model': str(form.get('model', ''))}

    @app.post('/api/v1/prediction', dependencies=auth)
    def legacy_predict(body: LegacyRequest, owner: Optional[str] = None):
        if owner is not None and owner != body.owner:
            raise HTTPException(422, 'Query owner and body owner must match')
        # Resolve both eyes before running either model to avoid partial bad requests.
        for image_id in body.image.values():
            if image_id:
                app.state.store.image(image_id)
        results, eyes = {'err_code': 0}, {}
        names = {'dr': 'DR', 'glaucoma': 'Glaucoma', 'amd': 'AMD', 'myopia': 'Myopia',
                 'cataract': 'Cataract', 'dme': 'DME'}
        for side, image_id in body.image.items():
            if not image_id:
                continue
            output = predict(PredictRequest(image_id=image_id, thresholds=body.thresholds, iqa_threshold=body.iqa_threshold))
            eyes[side] = output
            suffix = 'l' if side == 'left' else 'r'
            if 'qc' in body.models:
                results[f'QC_probability_{suffix}'] = output['quality']['quality_score']
            # Image quality is advisory: always return the requested disease heads.
            if not output['quality']['gradable']:
                results[f'ungradable_{suffix}'] = True
            for name in body.models:
                if name in names:
                    d = output['diseases'][name]
                    p, t = d['probability'], d['threshold']
                    # Existing model interfaces map the disease threshold onto 0.5.
                    results[f'{names[name]}_probability_{suffix}'] = (0.5+(p-t)/(1-t)*0.5) if p >= t else p/t*0.5
        response = {'id': uuid4().hex, 'owner': body.owner, 'patient_id': body.patient_id or None,
                    'patient_name': None, 'age': None, 'gender': None, 'ethnicity': None,
                    'ocular_history': None, 'diagnosis': None, 'recommendation': None,
                    'timestamp': int(time.time()), 'results': json.dumps(results, allow_nan=False),
                    'results_json': results, 'eyes': eyes}
        for side in ('left', 'right'):
            value = body.image.get(side, '')
            response[f'{side}_image_id'] = UUID(value).hex if value else ''
            response[f'{side}_image_filename'] = f'{UUID(value).hex}.png' if value else ''
        with app.state.store.db() as db:
            db.execute('INSERT INTO submissions VALUES (?,?)', (response['id'], json.dumps(response, allow_nan=False)))
        return response

    @app.get('/api/v1/prediction/{submission_id}', dependencies=auth)
    def get_legacy_prediction(submission_id: UUID):
        with app.state.store.db() as db:
            row = db.execute('SELECT result FROM submissions WHERE id=?', (submission_id.hex,)).fetchone()
        if not row:
            raise HTTPException(404, 'Submission not found')
        return json.loads(row[0])

    return app


app = create_app()
