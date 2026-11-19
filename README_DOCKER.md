# Fundus AI Docker deployment

## Disease naming

The JSON field `myopia` is displayed as **Pathologic myopia / 病理性近视**, matching the bilingual UI and English Word documents. Probability fields and threshold keys remain `myopia`; clients do not need a new field name.

Run the same REST API and bilingual web application on a CPU server or an NVIDIA GPU server. The image contains application code and Python dependencies; the two private model checkpoints are mounted read-only. Uploaded images and results are stored in a persistent Docker volume.

## Which API key belongs in the UI?

Enter the **exact value of `FUNDUS_API_KEY` configured on the running server**. Paste the value only: no `Bearer ` prefix, surrounding quotes or spaces. The browser sends it as `X-API-Key`.

- Existing local preview at `http://127.0.0.1:8056/`: `local-preview-key` was verified against the protected API. This credential is for that preview only.
- Docker deployment: use `FUNDUS_API_KEY` from your private `.env.docker` file, generated below. `local-preview-key` is intentionally too short for container startup.
- A fixed `FUNDUS_BEARER_TOKEN`, if configured separately through `docker run`, is for clients using the Authorization header; it is not the UI's X-API-Key value.
- `.env.api.example` does not configure a running process. Compose uses `.env.docker` only when `--env-file .env.docker` is supplied. Editing that file requires container recreation, not merely a browser refresh or `docker compose restart`.

Read your key locally with a text editor. Do not commit the file or include it in an image. The initializer creates it with mode 0600 and refuses to overwrite an existing key.

## 1 Prepare the server

Install Docker Engine and Docker Compose v2. Copy this deployment directory to the server, including:

```text
Dockerfile, .dockerignore, compose.yaml, compose.gpu.yaml
infer.py, fundus_api/, docker/, scripts/, samples/
weights/mermed6_ep41.pth
fundus_IQA/eyeqc_class_model4vit_base_lr1e-06-AUC9988-3DB-statedict.pth
```

The API and IQA weights must be real checkpoint files, not Git LFS pointer text. Their directories are mounted, not copied into image layers or the Docker build context. Original uploads, secrets, `.git` and documents are also excluded from the build context.

From the deployment root, generate the private environment file once:

```bash
python3 scripts/init_docker_env.py
```

Open `.env.docker` and save the `FUNDUS_API_KEY` value securely for your UI and API clients. An exported shell variable of the same name overrides the file in Compose; the client must use the effective server key. The default is loopback binding on port 5050. The application runs on port 8000 inside the container.

## 2 CPU deployment

No GPU or CUDA installation is required. On the target server:

```bash
docker compose --env-file .env.docker up -d --build --wait
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs --tail=100 api
```

The image is tagged `fundus-ai:cpu`; `FUNDUS_DEVICE=cpu` is explicit. The CPU build supports native Linux amd64 and arm64. Docker Desktop can run the CPU image on macOS; this does not provide Apple GPU/MPS access.

## 3 NVIDIA GPU deployment

Use a Linux x86_64 server with an NVIDIA GPU, a driver compatible with CUDA 12.1, and NVIDIA Container Toolkit configured for Docker. Run `nvidia-smi` on the host first. GPU configuration follows Docker's [GPU reservation documentation](https://docs.docker.com/compose/how-tos/gpu-support/) and NVIDIA's [Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose --env-file .env.docker \
  -f compose.yaml -f compose.gpu.yaml up -d --build --wait

docker compose --env-file .env.docker \
  -f compose.yaml -f compose.gpu.yaml exec -T api \
  python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The override builds `fundus-ai:gpu` with CUDA-enabled PyTorch, selects `FUNDUS_DEVICE=cuda`, and reserves one NVIDIA GPU. It uses Linux amd64 explicitly. It does not silently fall back to CPU if the GPU runtime is missing. CUDA user-space libraries come from the PyTorch wheel dependencies; the NVIDIA kernel driver and Container Toolkit belong on the host.

CPU and GPU use the same Compose service name and persistent volume. To switch mode, run the appropriate `up -d --build --wait` command; Compose recreates the service while preserving data. Do not run two copies against the same SQLite storage. GPU execution must be validated on actual NVIDIA hardware; a Mac CPU test is not GPU validation.

## 4 Test without the browser

In a client terminal, set TOKEN to the configured secret. With the default Compose configuration, it is the same value as `FUNDUS_API_KEY` in `.env.docker`.

```bash
export API_BASE=http://127.0.0.1:5050
export TOKEN='paste-your-FUNDUS_API_KEY-value-here'

curl --fail-with-body -sS "$API_BASE/health"

curl --fail-with-body -sS --max-time 120 \
  "$API_BASE/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@samples/FOJ-1000_L.jpg"
```

The sample should return `status: completed`, `quality.gradable: true`, six disease entries, `image_id` and `prediction_id`. API thresholds are retained; the main UI summary/table hides them, while the footer REST API response panel shows the complete JSON. `status: ungradable` and `quality.gradable: false` indicate a quality warning; all six disease predictions are still returned. The UI highlights the warning without hiding results. `/health` is public and checks readiness, not credentials; a protected endpoint is needed to verify your API key.

Use X-API-Key instead if desired:

```bash
curl --fail-with-body -sS --max-time 120 \
  "$API_BASE/api/v1/predict" \
  -H "X-API-Key: $TOKEN" \
  -F "file=@samples/FOJ-1000_L.jpg"
```

The image includes a smoke test, so the host only needs Docker. It uses the container's configured credential without printing it:

```bash
# CPU
# The resulting JSON is printed; the credential is not.
docker compose --env-file .env.docker exec -T api sh -c \
  'API_BASE=http://127.0.0.1:8000 TOKEN="$FUNDUS_API_KEY" bash scripts/smoke_test_api.sh'

# GPU: include the same override as at startup.
docker compose --env-file .env.docker \
  -f compose.yaml -f compose.gpu.yaml exec -T api sh -c \
  'API_BASE=http://127.0.0.1:8000 TOKEN="$FUNDUS_API_KEY" bash scripts/smoke_test_api.sh'
```

The script exits 0 when IQA and all six disease outputs are valid, including ungradable images. It prints a warning for ungradable images, and exits nonzero for request or validation failures. The container includes Bash and curl for this command. Alternatively run the host script with Python available: `TOKEN="$TOKEN" bash scripts/smoke_test_api.sh`.

Open `http://127.0.0.1:5050/` for the web UI, enter the server's key, and upload an image. Swagger and the OpenAPI schema are available at `/docs` and `/openapi.json`.

## 5 Expose the service to third parties

For an HTTPS reverse proxy on the same server, keep `FUNDUS_BIND_ADDRESS=127.0.0.1` and proxy to port 5050. Allow about 30 MiB at the proxy for compatible Base64 multipart requests, and set timeouts to match measured inference latency. Configure HTTPS and distribute the service key through your normal secret-sharing process.

If you need direct network access, edit `.env.docker` to set `FUNDUS_BIND_ADDRESS=0.0.0.0`, then recreate the container with the appropriate CPU/GPU startup command. Restrict access to the intended clients at the firewall. Use your server address in API_BASE; do not send patient images or service secrets over public unencrypted HTTP.

The current authentication model is one service credential. `owner` and `patient_id` are metadata and do not isolate tenants. Independent third parties requiring data isolation should use separate instances/volumes/secrets or an explicit tenant authorization layer.

## 6 Persistence and lifecycle

Docker Compose creates the named volume `fundus-ai_fundus-data` and mounts it at `/data`:

```text
/data/<image UUID>/original       uploaded bytes
/data/<image UUID>/image.png      normalized RGB inference image
/data/index.sqlite3              image index, predictions and submissions
```

Model mounts are `/app/weights` and `/app/fundus_IQA`, both read-only. The application runs as UID/GID 10001, not root. The named volume is initialized with writable ownership. If replacing it with a host bind mount, prepare that directory for UID 10001 before startup; the container does not change host permissions automatically.

```bash
# Stop and remove containers/network; retain the data volume.
docker compose --env-file .env.docker down

# Recreate after changing the secret or other environment settings.
docker compose --env-file .env.docker up -d --force-recreate --wait
```

Include `-f compose.yaml -f compose.gpu.yaml` for GPU lifecycle commands. Do not add `--volumes` to `down` if uploaded images/results must be kept. For an application-consistent backup, stop the service and back up the whole volume, including SQLite and image folders. No automated retention or deletion is implemented.

The health check allows two minutes for startup. `--wait` reports readiness but does not run disease inference; use the smoke test for that. Health checks do not automatically restart an unhealthy but still-running process; `restart: unless-stopped` restarts exited processes. Logs rotate at 10 MB with three files. One Uvicorn worker loads both models once; the model lock serializes inference. CPU thread defaults are 2; change `FUNDUS_CPU_THREADS` and measure latency and memory on the deployment host.

## 7 Build images or run without Compose

```bash
# CPU, native architecture of the Docker server.
docker build -t fundus-ai:cpu .

# Build for a typical x86_64 server from an ARM development machine.
docker buildx build --platform linux/amd64 --load -t fundus-ai:cpu .

# GPU, x86_64 Linux target.
docker buildx build --platform linux/amd64 --load \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t fundus-ai:gpu .
```

Example without Compose (run after creating `.env.docker`):

```bash
docker run -d --name fundus-ai-cpu --init \
  --restart unless-stopped \
  --env-file .env.docker -e FUNDUS_DEVICE=cpu \
  -p 127.0.0.1:5050:8000 \
  --mount type=bind,source="$PWD/weights",target=/app/weights,readonly \
  --mount type=bind,source="$PWD/fundus_IQA",target=/app/fundus_IQA,readonly \
  --mount type=volume,source=fundus-ai-data,target=/data \
  fundus-ai:cpu
```

For GPU, use the GPU image, `-e FUNDUS_DEVICE=cuda`, `--gpus all` and a different container name; stop the CPU instance first. The Docker-run example uses `fundus-ai-data`, while Compose uses `fundus-ai_fundus-data`; choose and retain one deployment method or explicitly reuse the same volume when migrating. Container health checks expect port 8000, so change only the host-side port mapping.

To transfer an existing image without a registry:

```bash
docker save -o fundus-ai-cpu.tar fundus-ai:cpu
# On a server with the matching CPU architecture:
docker load -i fundus-ai-cpu.tar
# Copy checkpoint folders and Compose configuration separately.
# Start without rebuilding the loaded image:
docker compose --env-file .env.docker up -d --no-build --wait
```

Images do not include checkpoints or uploaded data. Build separately for the server architecture; an ARM image cannot be treated as a native amd64 image. The GPU image can be saved/loaded similarly using its own tag and Compose override.

## 8 Versions and diagnostics

The Docker build uses Python 3.10, PyTorch 2.5.1 and torchvision 0.20.1, matching the model README's validated torch pair. CPU/CUDA wheel indexes follow [PyTorch's published installation matrix](https://pytorch.org/get-started/previous-versions/). Other top-level application versions are pinned in `docker/requirements.txt`; transitive dependencies and the base image are not fully locked by digest. Build and smoke-test a new image before adopting updates.

- Startup error about the key: run `python3 scripts/init_docker_env.py`, replace any example value with a private random key, and pass `--env-file .env.docker`. Do not use the short local preview key for the container.
- Missing checkpoint: check both host folders and file permissions; Compose rejects missing bind-mount directories.
- Permission denied under /data: use the named volume or give your bind directory the intended UID 10001 ownership.
- CUDA unavailable: check the GPU image tag, override flags, host driver and Container Toolkit. Diagnose host/runtime access before changing model code.
- HTTP 401: confirm the browser/client key and server environment match. Recreate after configuration changes; remove leading/trailing spaces from pasted values.
- Build-time certificate/proxy errors: configure the Docker daemon/build environment's approved proxy/CA. Do not disable certificate verification.

## Local validation record

Validated on 19 November 2026: the CPU image built successfully on Docker Desktop's Linux arm64 engine; the Compose container became healthy on host port 5050; a host request with no credential returned 401; the generated Docker key authenticated successfully; the real IQA and six-head smoke test passed; and a saved prediction remained identical after a container restart. All 17 API/container configuration unit tests passed. The merged GPU Compose configuration was checked for the CUDA wheel index, Linux amd64 platform, device reservation and read-only model mounts. Actual NVIDIA execution was not tested on this Mac; validate the GPU deployment on the target NVIDIA host using the commands above.


## Documentation consistency check

The English Word user guide and system design now include the same CPU/GPU Compose startup commands and external curl example used here. The default external address is `http://127.0.0.1:5050`; container-only smoke tests use `http://127.0.0.1:8000`. The legacy preview address `http://127.0.0.1:8056` and its short preview key are not Docker defaults. Patient uploads persist in `/data`, model mounts remain read-only, and ungradable images continue to return all six disease predictions.

The documentation review on 19 November 2026 checked all three READMEs and both English Word documents: disease display naming, 36 shell command blocks, CPU/GPU effective Compose configurations, CPU startup, both Word curl examples against the real Docker service, and the in-container smoke test. External port 5050, internal port 8000, credentials and advisory IQA behavior agree with the implementation. NVIDIA execution remains a target-server validation step.
