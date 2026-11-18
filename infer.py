#!/usr/bin/env python3
"""MerMED-6 眼底 6 病推理(DR / 青光眼 / AMD / 近视 / 白内障 / DME),单图、多图、文件夹都行。

用法:
  python infer.py a.jpg                      # 单图
  python infer.py a.jpg b.png c.jpg          # 多图
  python infer.py /path/to/folder            # 文件夹(递归找 jpg/png/jpeg/tif/bmp)
  python infer.py /path/to/folder -o out.csv # 结果写 csv
  python infer.py a.jpg --thr dr=0.4,dme=0.3 # 改某些病的阈值(其余用内置默认)
  python infer.py a.jpg --device cpu         # 没有 GPU

输出:每张图 6 个概率 p_<病>(各自独立的二分类 sigmoid 概率,不加到 1)和按阈值判的 有/无。
预处理与训练/评测完全一致:短边 LANCZOS 缩到 256 -> 中心裁 256 -> Resize(256, bicubic)
-> CenterCrop(224) -> 眼底专用 mean/std。输入应是彩色眼底照(单眼一张)。
"""
import argparse
import csv
import os
import sys
from functools import partial

import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms
from torchvision.transforms import InterpolationMode

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "weights", "mermed6_ep41.pth")
DISEASES = ["dr", "glaucoma", "amd", "myopia", "cataract", "dme"]   # 头的顺序,不可改
MEAN, STD = (0.33781426, 0.21333193, 0.13285545), (0.29643703, 0.1900187, 0.13929177)
THR = {"dr": 0.193, "glaucoma": 0.121, "amd": 0.017, "myopia": 0.010, "cataract": 0.292, "dme": 0.034}  # 见 README 第 4 节
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def build_model():
    return VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=len(DISEASES), drop_path_rate=0.2,
        global_pool="avg", dynamic_img_size=True)


def load_model(ckpt, device):
    model = build_model()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


_tf = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_image(path):
    """与训练缓存(cache_fundus5_256)一致:短边缩到 256 再中心裁 256,然后走 eval transform。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.LANCZOS)
    w2, h2 = im.size
    l, t = (w2 - 256) // 2, (h2 - 256) // 2
    return _tf(im.crop((l, t, l + 256, t + 256)))


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                out += [os.path.join(r, f) for f in sorted(fs) if f.lower().endswith(EXTS)]
        else:
            out.append(p)
    return out


class _DS(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            return load_image(self.files[i]), 1
        except Exception as e:                       # 坏图不让整批挂掉
            print(f"  !! 读不了 {self.files[i]}: {e}", file=sys.stderr)
            return torch.zeros(3, 224, 224), 0


@torch.no_grad()
def predict(model, files, device, batch_size=32, workers=4):
    """返回 [(path, {disease: prob} 或 None)],顺序与 files 一致。"""
    dl = torch.utils.data.DataLoader(_DS(files), batch_size=batch_size, num_workers=workers)
    res = []
    for x, ok in dl:
        p = torch.sigmoid(model(x.to(device)).float()).cpu().tolist()
        res += [row if o else None for row, o in zip(p, ok.tolist())]
    return [(f, dict(zip(DISEASES, r)) if r else None) for f, r in zip(files, res)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="图片文件和/或文件夹")
    ap.add_argument("-o", "--out", help="结果 csv 路径")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thr", default="", help="逐病阈值,如 dr=0.4,dme=0.3;默认用内置 THR")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    thr = dict(THR)
    for kv in filter(None, a.thr.split(",")):
        k, v = kv.split("=")
        assert k in thr, f"未知疾病 {k},可选 {DISEASES}"
        thr[k] = float(v)

    files = collect(a.inputs)
    if not files:
        sys.exit("没找到图片")
    model = load_model(a.ckpt, a.device)
    results = predict(model, files, a.device, a.batch_size, a.workers)

    print(f"{'image':40s} " + " ".join(f"{d:>9s}" for d in DISEASES))
    for f, p in results:
        name = f if len(f) <= 40 else "…" + f[-39:]
        if p is None:
            print(f"{name:40s} 读取失败")
            continue
        print(f"{name:40s} " + " ".join(f"{p[d]:7.3f}{'*' if p[d] >= thr[d] else ' '} " for d in DISEASES))
    print("(* = 概率 >= 阈值,判为有病;阈值: " + ", ".join(f"{d}={thr[d]:g}" for d in DISEASES) + ")")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image"] + [f"p_{d}" for d in DISEASES] + [f"pred_{d}" for d in DISEASES])
            for f, p in results:
                if p is None:
                    w.writerow([f] + [""] * 12)
                else:
                    w.writerow([f] + [f"{p[d]:.6f}" for d in DISEASES] + [int(p[d] >= thr[d]) for d in DISEASES])
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
