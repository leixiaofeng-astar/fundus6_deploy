# Fundus AI REST API 与演示网页

## Disease naming

The JSON field `myopia` is displayed as **Pathologic myopia / 病理性近视**, matching the bilingual UI and English Word documents. Probability fields and threshold keys remain `myopia`; clients do not need a new field name.

复用本目录 `infer.py` 的 MerMED-6 模型和 `fundus_IQA` 的 ViT 质量模型。包含 Word Quick Guide 兼容接口、直接上传推理接口、持久化图片与结果，以及默认英文、支持中英文切换的演示网页。原始 CLI 与权重不变。

## 启动

建议 Python 3.10+，使用已能运行原始模型的环境：

```bash
cd /Users/leixf/sunny/fundus6_deploy
python -m pip install -r requirements-api.txt
export FUNDUS_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# 默认 Bearer Token 与 API Key 相同；保存并安全分发这个随机值。
export FUNDUS_DEVICE=cpu
python -m uvicorn fundus_api.app:app --host 127.0.0.1 --port 5050 --workers 1 --limit-concurrency 8
```

打开 http://127.0.0.1:5050/，选择图片，在 API Key 输入框填写服务端 `FUNDUS_API_KEY`，点击上传并分析。Swagger 为 `/docs`，OpenAPI 为 `/openapi.json`。Swagger 的 Authorize 支持 Bearer 或 X-API-Key。浏览器不需要 npm、CDN 或前端构建。

环境变量：

| 变量 | 默认值 / 用途 |
|---|---|
| `FUNDUS_MODEL_ROOT` | 本部署根目录，包含 infer.py、weights、fundus_IQA |
| `FUNDUS_DEVICE` | 有 CUDA 时 cuda，否则 cpu |
| `FUNDUS_DATA_DIR` | `<部署根目录>/data/api` |
| `FUNDUS_API_KEY` | 未设置时为本地免认证演示；对外必须设置 |
| `FUNDUS_BEARER_TOKEN` | 默认等于 FUNDUS_API_KEY，可单独设置 |

`.env.api.example` 仅是示例，不自动读取。模型在启动时加载一次，加载失败则服务无法启动。模型调用由锁串行保护，CPU/GPU 都用一个 worker，避免重复占用模型内存。

## 接口

所有 `/api/v1/*` 接口使用 `Authorization: Bearer <token>` 或 `X-API-Key: <key>`。

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/v1/image` | Word 兼容：Base64 multipart，返回 `id` |
| POST | `/api/v1/prediction` | Word 兼容：左右眼 ID + models + owner + patient_id |
| GET | `/api/v1/prediction/{id}` | 取回兼容接口完整结果 |
| POST | `/api/v1/images` | 新接口：multipart 二进制 `file`，返回 `image_id` |
| GET | `/api/v1/images/{image_id}` | 图片元数据 |
| GET | `/api/v1/images/{image_id}/preview` | 已转换的 PNG，需认证 |
| POST | `/api/v1/predictions` | 新接口：单眼 ID 推理，可覆盖疾病及 IQA 阈值 |
| GET | `/api/v1/predictions/{prediction_id}` | 取回单眼结果 |
| POST | `/api/v1/predict/no-save` | 上传即推理，不持久保存图片/结果，不申请 ID |
| POST | `/api/v1/predict` | 一步上传 `file` 并返回模型输出，无需申请 ID |
| GET | `/health` | 模型加载成功后返回 ready |

### Word 兼容示例

```bash
export API_BASE=http://127.0.0.1:5050
export TOKEN="$FUNDUS_API_KEY"
export OWNER=partner-demo
# 上传两眼；不要手动构造 JSON 字符串
python - <<'PY'
import base64, json, os
from pathlib import Path
import httpx
base = os.environ['API_BASE']
owner = os.environ['OWNER']
with httpx.Client(headers={'Authorization': 'Bearer '+os.environ['TOKEN']}, timeout=120) as client:
    image_ids = {}
    for side, name in [('left', 'FOJ-1000_L.jpg'), ('right', 'FOJ-1000_R.jpg')]:
        path = Path('samples') / name
        fields = {'filename': (None, path.name), 'owner': (None, owner),
                  'content': (None, base64.b64encode(path.read_bytes()).decode())}
        response = client.post(base+'/api/v1/image', files=fields)
        response.raise_for_status()
        image_ids[side] = response.json()['id']
    response = client.post(base+'/api/v1/prediction', params={'owner': owner}, json={
        'image': image_ids, 'models': ['qc','cataract','glaucoma','dr','amd','dme','myopia'],
        'patient_id': '', 'owner': owner})
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
PY
```

`/image` 接收 Word 中的 `filename`、`content`、`owner`，可选 `name`、`model`；`model` 是上传元数据，不会自动选择预测模型。上传时返回 32 位十六进制 `id`，新接口也接受这种 ID。`image` 支持 left、right 任意一眼或双眼；空的另一眼可省略。

兼容接口返回 `left_image_id`、`right_image_id`、`patient_id`、`owner`、`timestamp` 等字段。`results` 保持 Word 示例中的字符串类型，但内容为合法 JSON，请用 `json.loads`；新增 `results_json` 对象和 `eyes` 完整单眼结果，避免任何 eval。

兼容概率字段如 `DR_probability_l`、`Glaucoma_probability_r` 按参考项目函数惯例，把疾病阈值分段线性映射到 0.5，方便旧客户端以 0.5 判断阳性。它们不是原始 sigmoid 概率，也不是医学概率校准。`QC_probability_l/r` 是**越高越好**的质量分数。原始疾病概率、实际阈值和布尔判定位于 `eyes.left/right.diseases`；原始**低质量**概率位于 `eyes.left/right.quality.poor_quality_probability`。

低质量时仍继续六病推理并返回 HTTP 200，`eyes.<side>.status=ungradable`、`quality.gradable=false`，`diseases` 和旧疾病概率字段仍包含预测。`results.err_code=0` 表示推理成功，`ungradable_l/r=true` 单独标记质量警示；这些结果需结合低质量警示解读。未知 models 返回 422，不静默跳过。当前仅提供 qc 和六病模型。

这里没有患者注册、患者数据库或原平台的 OAuth/JWT 签发服务。`patient_id` 作为外部标识回传，可为空；患者姓名等字段为 null。Bearer 使用本服务配置的固定随机秘密，不验证旧服务器签发的 JWT，也不使用文档中的测试 token。兼容接口的预览文件名是展示标识，真实预览走受保护的 `/images/{id}/preview`。

### 一次上传直接推理

```bash
curl --fail-with-body -sS "$API_BASE/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@samples/FOJ-1000_L.jpg"
```

服务仍为该图片自动生成并返回 image_id，将原始文件和转换后的 PNG 保存；客户端不用提前申请 ID。新接口输出示意（数值为示例）：

```json
{
  "prediction_id": "<uuid>", "image_id": "<uuid>", "status": "completed",
  "quality": {"poor_quality_probability": 0.01, "threshold": 0.67,
              "quality_score": 0.9925, "gradable": true},
  "diseases": {"dr": {"probability": 0.6, "threshold": 0.193, "positive": true}},
  "model_version": "mermed6_ep41", "iqa_version": "eyeqc_vit_base_AUC9988_3DB",
  "created_at": "<UTC ISO8601>"
}
```

实际成功结果包含六病；示例仅展开 DR。各病概率独立，不相加为 1。

### 分两步上传与推理

```bash
curl --fail-with-body -sS "$API_BASE/api/v1/images" \
  -H "Authorization: Bearer $TOKEN" -F "file=@samples/FOJ-1000_L.jpg"
# 用返回的 image_id 替换下值：
curl --fail-with-body -sS "$API_BASE/api/v1/predictions" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"image_id":"<image_id>","thresholds":{"dr":0.193},"iqa_threshold":0.67}'
```

阈值必须有限且严格在 (0,1) 内。未覆盖疾病沿用 infer.py 的 THR。直接上传接口使用默认阈值；需要自定义时使用两步接口。

## Python 模型接口

```python
from fundus_api.models import load_dr_model, get_dr_ai_prediction
model = load_dr_model(device='cpu')
positive, raw_probability = get_dr_ai_prediction('samples/FOJ-1000_L.jpg', model, 0.193, 'cpu')

from fundus_api.models import ModelService
service = ModelService('cpu')
result = service.predict('samples/FOJ-1000_L.jpg', {}, 0.67)
```

`load_fundus_model` / `get_fundus_ai_prediction` 提供六病接口；`load_iqa_model` / `get_iqa_prediction` 提供 IQA 接口；需要质量检查和并发保护时用 `ModelService`。DR 函数返回原始概率，不返回参考项目的五分类 DR 分级。六病预处理与原始 CLI 一致。

IQA 保留 `fundus_IQA/test_iqa.py` 的 ViT 包装、ImageNet 标准化、224×224 Resize 和末类 softmax。用本地完整权重构建，不下载预训练权重。默认低质量阈值为原函数默认值 0.67，可在请求中显式设 0.76 等其他值。质量分数是原函数分段映射后的 `1-probability_1`，不是额外训练的模型。推理错误抛出异常；HTTP 500 不会伪造阴性结果。

## 图像、存储和部署边界

- 输入 JPG、PNG、BMP、单帧 TIFF，以及单帧 DICOM；原始文件上限 20 MiB、像素上限 2500 万。请求整体大小在 multipart 解析前限额，Base64 允许约 29 MiB 请求体；无 Content-Length 的分块请求也受限。
- DICOM 支持 8 位 RGB/YBR 和 MONOCHROME1/2（窗变换后转换）；多帧及不支持的像素布局返回 422。压缩 DICOM 可能需要安装 `pylibjpeg`/`pylibjpeg-libjpeg`/`pylibjpeg-openjpeg`。灰度文件可解码不代表模型已在该图像域验证。
- 存储：`data/api/<UUID>/<原文件名主体>__original.<原后缀>` 保留上传原文件，`<原文件名主体>__image.png` 用于推理和预览，`index.sqlite3` 保存图片索引、单眼结果及双眼提交结果。转换 PNG 不带原始 DICOM 患者标签，原始文件仍含原有信息。存储不是公开静态目录。
- 认证是**单租户服务凭证**。owner 与 patient_id 是业务元数据，不能作为用户身份或多租户隔离机制。不同第三方若需严格隔离，应使用独立实例/存储/凭证，或增加账号与租户权限层。
- 原文件和结果会持续保留，不自动删除。上线应按实际约定配置数据目录备份、磁盘限额、保留期和清理策略。
- 给第三方使用时，在反向代理后启用 HTTPS、认证、请求速率及连接限制；只有需要远程访问时才将 host 改为 0.0.0.0。建议代理限制请求体 30 MiB。当前交付是本地实现与演示，未部署到公网。
- CPU 推理按锁串行；长时间批量任务、高并发、队列、计费、账号管理均不在这个简单服务内。
- 保留原模型 README 的研究用途和限制；网页仅显示模型输出，不生成临床诊断建议。

## 验证

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python tests/verify_real_models.py
# 可选与原始 CLI 对比：
python infer.py samples/ -o /tmp/fundus-baseline.csv --device cpu --workers 0
FUNDUS_BASELINE_CSV=/tmp/fundus-baseline.csv PYTHONPATH=. python tests/verify_real_models.py
```

单元测试使用可注入模型，覆盖认证、兼容双眼流程、真实格式 DICOM 解码、输入校验、坏图清理、上传限额、低质量响应、模型异常、持久化和网页路由。`verify_real_models.py` 加载真实六病及 IQA 权重，验证真实兼容和一步上传 API，另验证低质量时仍调用疾病模型并保留六病结果，并可与原始 CLI CSV 比较（允许误差 2e-6）。


## English Word documentation and bilingual UI

- User guide: [Fundus AI API User Guide](docs/Fundus_AI_API_User_Guide.docx).
- Design and deployment: [Fundus AI System Design](docs/Fundus_AI_System_Design.docx).
- The web UI starts in English on every page load; use the Language menu for Chinese. Switching languages preserves results and does not rerun inference.
- Quality and disease thresholds remain hidden in the main UI summary and table. The footer REST API response control expands the complete response JSON, including threshold fields.
- To verify the real service without the web UI, run from the deployment root with the model Python environment activated:

```bash
export API_BASE=http://127.0.0.1:5050
export TOKEN='the-configured-service-credential'
bash scripts/smoke_test_api.sh
```

The script checks readiness, authenticated file upload, IQA, all six disease heads, and decision consistency. Exit 0 means IQA and all six disease outputs passed validation. Ungradable images still return all six outputs and print a quality warning; request or validation failures return a nonzero exit. Set IMAGE to use another image and PYTHON to select a Python executable if required.

## Docker deployment and API keys

CPU and NVIDIA GPU Docker deployment, persistent storage, image build commands, and curl tests are documented in [README_DOCKER.md](README_DOCKER.md). Create the private Docker configuration with `python3 scripts/init_docker_env.py`; then use `docker compose --env-file .env.docker up -d --build --wait` for CPU, or add `-f compose.yaml -f compose.gpu.yaml` for GPU.

```bash
python3 scripts/init_docker_env.py
# CPU
docker compose --env-file .env.docker up -d --build --wait
# NVIDIA GPU server instead of CPU
docker compose --env-file .env.docker \
  -f compose.yaml -f compose.gpu.yaml up -d --build --wait
```

The web UI API key is the exact `FUNDUS_API_KEY` configured on that server, without a `Bearer ` prefix. For Docker, read the value in the private `.env.docker` file. The existing temporary 8056 preview uses `local-preview-key`; that is a separate service credential and should not be used for the Docker deployment.


## Advisory image quality and raw response

Image quality no longer blocks disease prediction. New ungradable results have `status=ungradable`, `quality.gradable=false`, and six populated disease predictions. The UI highlights the warning and preview, and keeps the prediction table visible. The footer REST API response button shows the exact JSON from the latest completed request without rerunning inference. Existing saved results from older service versions are preserved; re-run an old image ID to produce predictions under the new behavior.


## Upload filenames and no save inference

Image IDs still use UUID v4. New persisted uploads keep the original name in their filenames, for example:

```text
<data root>/<image UUID>/
  2026-08-04-14-01-20-722_OS__original.jpg
  2026-08-04-14-01-20-722_OS__image.png
```

`filename` is the source basename. `stored_original_filename` and `stored_image_filename` report the actual saved names. Path separators/unsafe characters are sanitized and long names are shortened. Existing UUID directories and older `original` / `image.png` files remain supported; existing IDs are unchanged and files are not renamed retrospectively.

`POST /api/v1/predict` still saves the image and result. For **no persistent image or result storage**, use the new endpoint:

```bash
export API_BASE=http://18.138.92.89:8056
export TOKEN='your-server-credential'
curl --fail-with-body -sS --max-time 120 \
  "$API_BASE/api/v1/predict/no-save" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./2026-08-04-14-01-20-722_OS.jpg"
```

This is one upload/inference request, with no image-ID application. The response includes `saved: false`, `image_id: null`, `prediction_id: null`, the source `filename`, IQA and the six disease outputs. There is no database insert and no result-retrieval ID. Models and quality-warning behavior are the same as the persisted endpoint. The API uses default thresholds; authentication, 20 MiB upload limits, DICOM decoding and 25-megapixel limits still apply.

Temporary working files (and multipart spooling when required) may be written to the server's temporary directory. They are cleaned at the end of the request, including decoding/inference failures. This is not a RAM-only processing guarantee; it is a no-retention endpoint. As with normal temporary files, an OS crash or forced process termination cannot guarantee immediate cleanup.

A ready-to-run script combines the entire operation:

```bash
API_BASE=http://18.138.92.89:8056 TOKEN='your-server-credential' \
  bash scripts/predict_no_save.sh ./2026-08-04-14-01-20-722_OS.jpg
```

The remote examples require the updated server deployment. For local Docker, set API_BASE=http://127.0.0.1:5050 instead. Rebuild/recreate the container after copying the updated code to the server:

```bash
docker compose --env-file .env.docker up -d --build --wait
# NVIDIA GPU alternative:
docker compose --env-file .env.docker \
  -f compose.yaml -f compose.gpu.yaml up -d --build --wait
```

