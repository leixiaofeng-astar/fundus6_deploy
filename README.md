# MerMED-6 眼底 6 病推理

DR / 青光眼 / AMD / 病理性近视（Pathologic myopia） / 白内障 / DME,每个病一个独立二分类头(多标签,概率不加到 1)。
模型:ViT-B/16(MerMED-FM 微调),权重 `weights/mermed6_ep41.pth`(= fundus5 的 mermed_v3_dme6/checkpoint-best.pth,ep41)。

## 1. 环境
```bash
conda create -n mermed6 python=3.10 -y && conda activate mermed6
# GPU(按本机 CUDA 选 index-url,这里以 cu121 为例);只用 CPU 可去掉 --index-url
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
已验证版本:torch 2.5.1 / torchvision 0.20.1 / timm 1.0.28 / pillow 12 / numpy 2.4(timm ≥ 1.0 即可)。
显存:推理 batch 32 约 2–3 GB;CPU 也能跑,约 0.3–1 s/张。不依赖 MerMED 仓库。

## 2. 用法
```bash
python infer.py samples/FOJ-1000_L.jpg                    # 单图
python infer.py samples/FOJ-1000_L.jpg samples/FOJ-1000_R.jpg   # 多图
python infer.py samples/ -o out.csv                       # 文件夹,结果存 csv
python infer.py img.jpg --thr dr=0.4,dme=0.3              # 改某些病的阈值(默认用内置值,见第 4 节)
python infer.py samples/ --device cpu
```
Python 里调用:
```python
from infer import load_model, predict
model = load_model("weights/mermed6_ep41.pth", "cuda")
for path, probs in predict(model, ["a.jpg", "b.jpg"], "cuda"):
    print(path, probs)   # {"dr":0.03,"glaucoma":...,"dme":...}
```

## 3. 输入与输出
- 输入:彩色眼底照,**一张一眼**,jpg/png 等,任意尺寸(内部缩到短边 256 再中心裁剪)。
- 输出:`p_dr p_glaucoma p_amd p_myopia p_cataract p_dme`,各自 0–1 概率;csv 另含 `pred_<病>`(0/1)。

## 4. 使用须知
- **默认阈值(内置于 infer.py 的 `THR`)**:每病一个固定阈值,取使 min(Acc, Sen, Spe) 最大的点,
  在 SIDRP 训练半边上定(本模型没训练过 SIDRP),原样用到 SIDRP 测试半边和 SEERS 上报告:

  | 疾病 | 阈值 | SIDRP test Acc/Sen/Spe | SEERS Acc/Sen/Spe |
  |---|---|---|---|
  | DR (≥2) | 0.193 | 0.844 / 0.848 / 0.843 | 0.754 / 0.748 / 0.754 |
  | 青光眼 | 0.121 | 0.768 / 0.752 / 0.769 | 0.640 / 0.833 / 0.624 |
  | AMD (≥2) | 0.017 | 0.764 / 0.781 / 0.763 | 0.641 / 0.775 / 0.625 |
  | 白内障 | 0.292 | 0.796 / 0.823 / 0.795 | 0.621 / 0.603 / 0.636 |
  | DME | 0.034 | 0.719 / 0.762 / 0.718 | 0.632 / 0.857 / 0.628 |
  | 病理性近视（myopia；原评估 SE≤-6） | 0.010 | 0.671 / 0.671 / 0.672(UKBB,样本内) | — |

  取 min 最大意味着 Sen ≈ Spe,低患病率下 Acc 会比"全判阴性"低(全判阴性在 SIDRP 上 Acc 0.96–0.98 但 Sen=0,没有意义)。
  阈值依赖人群患病率:换到高/低患病率场景(如筛查 vs 转诊)请重新校准,或用 `--thr` 改。
  计算脚本与完整结果:fundus5/scripts/fix_thr.py、fundus5/doc/fixed_thr.md。
- **DR / AMD**:模型只给"有没有",不分级;SIDRP 评测口径是 ≥2 级。
- **病理性近视（Pathologic myopia；JSON 字段: `myopia`）**：名称与 UI 和文档一致。原性能表的 myopia 评估依据为 UKBB SE≤-6，保留该技术口径；此次名称同步沿用原权重和阈值。
- **DME**:眼底照上的 DME 本身较难,建议结合 OCT。
- 仅供研究,不是诊断/医疗器械。

## REST API and Docker deployment

The six-disease CLI is also available through a REST API and an English/Chinese web UI. See [README_API.md](README_API.md) for the API contract and [README_DOCKER.md](README_DOCKER.md) for CPU/GPU Docker deployment, API-key configuration, persistent storage and curl smoke tests.


## Docker quick start and API test

Run from the deployment root. CPU and GPU are alternative modes on the same default host port 5050. Generate the private Docker env file once, then select one startup command:

```bash
python3 scripts/init_docker_env.py
# CPU
docker compose --env-file .env.docker up -d --build --wait
# NVIDIA GPU server instead of CPU
docker compose --env-file .env.docker \
  -f compose.yaml -f compose.gpu.yaml up -d --build --wait
```

Use the configured `FUNDUS_API_KEY` as TOKEN; the UI takes the key value without a Bearer prefix. The JSON field `myopia` is displayed as **Pathologic myopia / 病理性近视**. The separate 8056 preview uses a different credential.

```bash
export API_BASE=http://127.0.0.1:5050
export TOKEN='paste-your-FUNDUS_API_KEY-value-here'
curl --fail-with-body -sS --max-time 120 \
  "$API_BASE/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@samples/FOJ-1000_L.jpg"
```

See [README_DOCKER.md](README_DOCKER.md) for requirements, GPU validation, persistent volumes and lifecycle commands. Ungradable images still return all six disease predictions with a quality warning.


## File names and transient inference

New saved files include the sanitized source filename in their UUID directory. UUID generation is unchanged; older image records remain readable. Use `POST /api/v1/predict/no-save` or `scripts/predict_no_save.sh` to upload once and receive all predictions without retaining uploaded images or results. Temporary processing files are cleaned after each request. See [README_API.md](README_API.md#upload-filenames-and-no-save-inference) for examples and storage details.
