"""Public model interface. Errors raise exceptions, never synthetic predictions."""
import math
import os
from pathlib import Path
from threading import Lock
import torch
from PIL import Image
from torch import nn
from timm import create_model
from torchvision import transforms
from .imaging import read_image

ROOT = Path(os.getenv('FUNDUS_MODEL_ROOT', Path(__file__).resolve().parents[1]))
DISEASES = ('dr', 'glaucoma', 'amd', 'myopia', 'cataract', 'dme')
THRESHOLDS = {'dr': 0.377, 'glaucoma': 0.458, 'amd': 0.052, 'myopia': 0.063, 'cataract': 0.670, 'dme': 0.177}


def threshold_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError('Threshold must be finite and strictly between 0 and 1')
    return value


def load_fundus_model(model_path=None, device='cpu'):
    # Reuse the supplied architecture, weights and exact training preprocessing.
    import importlib.util
    spec = importlib.util.spec_from_file_location('fundus6_infer', ROOT / 'infer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_model(str(model_path or ROOT / 'weights/mermed6_ep41.pth'), device)
    model._fundus_preprocess = module._tf
    return model


def get_fundus_ai_prediction(image_path, model, thresholds=None, device='cpu'):
    thresholds = thresholds or {}
    if set(thresholds) - set(DISEASES):
        raise ValueError('Unknown disease threshold')
    limits = {k: threshold_value(thresholds.get(k, v)) for k, v in THRESHOLDS.items()}
    im = read_image(image_path)
    w, h = im.size
    scale = 256 / min(w, h)
    im = im.resize((max(256, round(w * scale)), max(256, round(h * scale))), Image.Resampling.LANCZOS)
    w, h = im.size
    im = im.crop(((w-256)//2, (h-256)//2, (w-256)//2+256, (h-256)//2+256))
    with torch.inference_mode():
        probs = model(model._fundus_preprocess(im).unsqueeze(0).to(device)).float().sigmoid()[0].cpu().tolist()
    if len(probs) != 6 or not all(math.isfinite(p) for p in probs):
        raise RuntimeError('Invalid disease model output')
    return {k: {'probability': float(p), 'threshold': limits[k], 'positive': p >= limits[k]}
            for k, p in zip(DISEASES, probs)}


def load_dr_model(model_path=None, device='cpu'):
    """DR uses the DR head of the supplied six-disease model."""
    return load_fundus_model(model_path, device)


def get_dr_ai_prediction(image_path, model, threshold=THRESHOLDS['dr'], device='cpu'):
    result = get_fundus_ai_prediction(image_path, model, {'dr': threshold}, device)['dr']
    return int(result['positive']), result['probability']


class IQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Same wrapper and state-dict keys as fundus_IQA/test_iqa.py.
        # Local checkpoint supplies all weights; never download pretrained weights.
        self.model = create_model('vit_base_patch16_224', pretrained=False, num_classes=2)

    def forward(self, x):
        return self.model(x)


def load_iqa_model(model_path=None, device='cpu'):
    model = IQAModel()
    path = model_path or ROOT / 'fundus_IQA/eyeqc_class_model4vit_base_lr1e-06-AUC9988-3DB-statedict.pth'
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval()


_iqa_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_iqa_prediction(image_path, model, threshold=0.67, device='cpu'):
    threshold = threshold_value(threshold)
    with torch.inference_mode():
        poor = float(model(_iqa_transform(read_image(image_path)).unsqueeze(0).to(device)).softmax(1)[0, -1].cpu())
    if not math.isfinite(poor):
        raise RuntimeError('Invalid IQA model output')
    adjusted = 0.5 + (poor-threshold)/(1-threshold)*0.5 if poor >= threshold else poor/threshold*0.5
    return {'poor_quality_probability': poor, 'threshold': threshold,
            'quality_score': 1-adjusted, 'gradable': poor < threshold}


class ModelService:
    def __init__(self, device=None):
        self.device = device or os.getenv('FUNDUS_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = Lock()
        self.fundus = load_fundus_model(device=self.device)
        self.iqa = load_iqa_model(device=self.device)

    def predict(self, path, thresholds, iqa_threshold):
        with self.lock:
            quality = get_iqa_prediction(path, self.iqa, iqa_threshold, self.device)
            diseases = get_fundus_ai_prediction(path, self.fundus, thresholds, self.device)
        return {'status': 'completed' if quality['gradable'] else 'ungradable',
                'quality': quality, 'diseases': diseases, 'model_version': 'mermed6_ep41',
                'iqa_version': 'eyeqc_vit_base_AUC9988_3DB'}
