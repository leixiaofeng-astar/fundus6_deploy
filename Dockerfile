# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FUNDUS_MODEL_ROOT=/app \
    FUNDUS_DATA_DIR=/data \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 bash curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 fundus \
    && useradd --uid 10001 --gid fundus --no-create-home fundus \
    && mkdir -p /data /app/weights /app/fundus_IQA \
    && chown fundus:fundus /data
COPY docker/requirements.txt /tmp/requirements.txt
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir --upgrade pip==25.3 \
    && python -m pip install --no-cache-dir --only-binary=:all: \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check
COPY infer.py ./infer.py
COPY fundus_api/ ./fundus_api/
COPY samples/ ./samples/
COPY scripts/smoke_test_api.sh ./scripts/smoke_test_api.sh
COPY docker/entrypoint.py /app/docker-entrypoint.py
# Import checks run at build time without loading checkpoints or accessing GPUs.
RUN python -c "import torch, torchvision, timm, pydicom; from fundus_api.app import app; assert app"
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]
ENTRYPOINT ["python", "/app/docker-entrypoint.py"]
CMD ["python", "-m", "uvicorn", "fundus_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8"]
