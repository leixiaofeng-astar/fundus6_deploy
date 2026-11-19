"""Container-only startup validation. Never print configured credentials."""
import os
from pathlib import Path
import sys


def validate(environ):
    key = environ.get('FUNDUS_API_KEY', '')
    if len(key.strip()) < 24 or key in ('replace-with-a-long-random-secret', 'set-a-private-random-secret'):
        raise ValueError('Set FUNDUS_API_KEY to a private random secret of at least 24 characters. Enter this same value in the web UI API key field.')
    bearer = environ.get('FUNDUS_BEARER_TOKEN', key)
    if bearer and len(bearer.strip()) < 24:
        raise ValueError('FUNDUS_BEARER_TOKEN must contain at least 24 characters when configured.')
    device = environ.get('FUNDUS_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Set FUNDUS_DEVICE to cpu or cuda.')
    model_root = Path(environ.get('FUNDUS_MODEL_ROOT', '/app'))
    for relative in ('weights/mermed6_ep41.pth', 'fundus_IQA/eyeqc_class_model4vit_base_lr1e-06-AUC9988-3DB-statedict.pth'):
        path = model_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing model checkpoint: '+str(path)+'. Mount the weights and fundus_IQA folders read-only.')
    directory = Path(environ.get('FUNDUS_DATA_DIR', '/data'))
    directory.mkdir(parents=True, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise ValueError('Data directory is not writable by UID 10001: '+str(directory))
    return device


def main():
    try:
        device = validate(os.environ)
        os.environ['FUNDUS_DEVICE'] = device
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable. Use the GPU image with --gpus or the GPU Compose override, on an NVIDIA Container Toolkit host.')
        if len(sys.argv) < 2:
            raise ValueError('Missing server command')
    except (ValueError, OSError) as error:
        print('Container configuration error: '+str(error), file=sys.stderr)
        raise SystemExit(1)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()
