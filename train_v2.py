# ================================================================
# HiSMD-Net Ablation Study — train_v2.py
# ================================================================
# COMPLETELY STANDALONE — does NOT touch train.py, hismdet_output/,
# app.py, index.html, README.md, or any other existing file.
#
# 7 ABLATION VARIANTS:
#   v1 : Full HiSMD-Net (ASRFR + SpatialSSM + Stride-2 head)
#   v2 : Without ASRFR
#   v3 : Without SpatialSSM
#   v4 : Without Stride-2 head  (strides: 4, 8, 16 only)
#   v5 : Without ASRFR + SpatialSSM
#   v6 : Without ASRFR + Stride-2 head
#   v7 : Without SpatialSSM + Stride-2 head
#
# USAGE:
#   python train_v2.py --variant all          # runs v1..v7 sequentially
#   python train_v2.py --variant 1            # runs only v1
#   python train_v2.py --variant 3 5          # runs v3 and v5
#   python train_v2.py --variant 2 --epochs 50
#
# Outputs:  ablation_output/v{1..7}_{name}/
#             training_log.csv   best_v{N}.pt   last_v{N}.pt
#           ablation_output/ablation_comparison.csv
#
# Bug fix vs. original train.py:
#   The regression loss used `pb[pos]` where pb was computed inside
#   a torch.no_grad() block, making the tensor non-differentiable.
#   Fixed: pb for the IoU loss is now re-decoded OUTSIDE no_grad(),
#   so gradients flow from the ciou_loss back to reg_heads.
# ================================================================

import os
import sys
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import json
import math
import time
import random
import glob
import warnings
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings("ignore")

# ── HPC package bootstrap ─────────────────────────────────────────────────
MY_PACKAGES = os.path.expanduser("~/my_python_packages")
if os.path.exists(MY_PACKAGES) and MY_PACKAGES not in sys.path:
    sys.path.insert(0, MY_PACKAGES)


def ensure_package(import_name, pip_name):
    """Ensure optional dependencies are installed on HPC user environment."""
    if "-h" in sys.argv or "--help" in sys.argv:
        return
    try:
        __import__(import_name)
    except ImportError:
        os.makedirs(MY_PACKAGES, exist_ok=True)
        if MY_PACKAGES not in sys.path:
            sys.path.insert(0, MY_PACKAGES)
        print(f"[+] Installing {pip_name} to {MY_PACKAGES} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--target", MY_PACKAGES, "--upgrade", pip_name],
                                  timeout=120)
        except Exception as e:
            print(f"[!] Warning: automatic installation of {pip_name} failed: {e}")


ensure_package("pycocotools", "pycocotools")
ensure_package("torchvision",  "torchvision")

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torchvision.ops import nms, box_iou

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None
from tqdm import tqdm

# ================================================================
# 0. VARIANT REGISTRY
# ================================================================

VARIANT_CONFIGS: Dict[int, dict] = {
    1: {"name": "full",               "use_asrfr": True,  "use_ssm": True,  "use_stride2": True },
    2: {"name": "no_asrfr",           "use_asrfr": False, "use_ssm": True,  "use_stride2": True },
    3: {"name": "no_ssm",             "use_asrfr": True,  "use_ssm": False, "use_stride2": True },
    4: {"name": "no_stride2",         "use_asrfr": True,  "use_ssm": True,  "use_stride2": False},
    5: {"name": "no_asrfr_no_ssm",    "use_asrfr": False, "use_ssm": False, "use_stride2": True },
    6: {"name": "no_asrfr_no_stride2","use_asrfr": False, "use_ssm": True,  "use_stride2": False},
    7: {"name": "no_ssm_no_stride2",  "use_asrfr": True,  "use_ssm": False, "use_stride2": False},
}

# ================================================================
# 1. CLI
# ================================================================

parser = argparse.ArgumentParser(description="HiSMD-Net Ablation Study (train_v2.py)")
parser.add_argument("--variant", nargs="+", default=["all"],
                    help="Variant IDs to run: 1-7, or 'all' (default: all)")
parser.add_argument("--epochs",     type=int,   default=30,
                    help="Epochs per variant (default: 30)")
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--batch_size", type=int,   default=0,
                    help="0 = auto from VRAM")
parser.add_argument("--lr",         type=float, default=5e-4)
parser.add_argument("--eval_freq",  type=int,   default=5,
                    help="Evaluate every N epochs (default: 5)")
parser.add_argument("--no_vis",     action="store_true",
                    help="Skip per-epoch visualisations (faster)")
args = parser.parse_args()

# Resolve which variants to run
if args.variant == ["all"]:
    VARIANTS_TO_RUN = list(VARIANT_CONFIGS.keys())
else:
    VARIANTS_TO_RUN = []
    for v in args.variant:
        vid = int(v)
        if vid not in VARIANT_CONFIGS:
            raise ValueError(f"Invalid variant id: {vid}. Must be 1-7.")
        VARIANTS_TO_RUN.append(vid)

EPOCHS     = args.epochs
SEED       = args.seed
EVAL_FREQ  = args.eval_freq

# ================================================================
# 2. GLOBAL CONFIGURATION
# ================================================================

IMG_SIZE     = 640
NUM_CLASSES  = 10
CLASS_NAMES  = ["pedestrian","people","bicycle","car","van",
                "truck","tricycle","awning-tricycle","bus","motor"]
COLORS       = [(255,64,64),(255,178,64),(64,255,64),(64,178,255),
                (178,64,255),(255,64,178),(64,255,178),(255,255,64),
                (255,128,0),(64,64,255)]
NECK_CH      = 128
ASRFR_CH     = 32
MIN_LR       = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 10.0
W_CLS        = 1.0
W_BOX        = 4.0
W_L1         = 1.0
CONF_THRESH  = 0.05
VIS_CONF     = 0.25
NMS_IOU      = 0.50
MAX_DET      = 300

USE_AMP = torch.cuda.is_available()
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

_vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
         if torch.cuda.is_available() else 0)
BATCH_SIZE = (args.batch_size if args.batch_size > 0
              else (16 if _vram > 50 else (8 if _vram > 20 else (4 if torch.cuda.is_available() else 2))))
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# Output root — sibling to this script, never inside hismdet_output/
THIS_DIR    = Path(__file__).parent
ABLATION_DIR = THIS_DIR / "ablation_output"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s: int):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ================================================================
# 3. DATASET UTILS (identical to original train.py)
# ================================================================

def find_visdrone_split(split_name: str) -> Optional[str]:
    roots = [
        THIS_DIR / "Dataset",
        THIS_DIR,
        Path(os.path.expanduser("~/Dataset")),
        Path(os.path.expanduser("~/uavdet_system/uavdet-system/Dataset")),
    ]
    exact = [
        THIS_DIR / "Dataset" / split_name / split_name,
        THIS_DIR / "Dataset" / split_name,
    ]
    for sd in exact:
        if (sd / "images").is_dir() and (sd / "annotations").is_dir():
            return str(sd.resolve())
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            for sd in [root / split_name] + list(root.rglob(split_name)):
                sd = sd.resolve()
                if sd in seen:
                    continue
                seen.add(sd)
                if sd.is_dir() and (sd / "images").is_dir() and (sd / "annotations").is_dir():
                    return str(sd)
        except Exception:
            pass
    return None


TRAIN_SPLIT = find_visdrone_split("VisDrone2019-DET-train") or str(
    THIS_DIR / "Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train")
VAL_SPLIT   = find_visdrone_split("VisDrone2019-DET-val") or str(
    THIS_DIR / "Dataset/VisDrone2019-DET-val")


class VisDroneDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

    def __init__(self, img_dir, ann_dir, img_size=640, augment=True, max_images=None):
        self.img_dir  = Path(img_dir)
        self.ann_dir  = Path(ann_dir)
        self.img_size = img_size
        self.augment  = augment
        paths = sorted([f for f in self.img_dir.iterdir()
                        if f.is_file() and f.suffix in self.EXTS])
        if not paths:
            paths = sorted([f for f in self.img_dir.rglob("*")
                            if f.is_file() and f.suffix in self.EXTS])
        self.img_paths = paths[:max_images] if max_images else paths
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"  Dataset [{img_dir}]: {len(self.img_paths)} images "
              f"({'train' if augment else 'val'})")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        img = cv2.imread(str(ip))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        boxes, cls_ids = self._load_ann(ip.stem, w0, h0)
        if self.augment:
            img, boxes, cls_ids = self._augment(img, boxes, cls_ids, w0, h0)
        img, boxes, pad_l, pad_t, scale = self._letterbox(img, boxes)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1))
        s = self.img_size
        target = torch.zeros((0, 5), dtype=torch.float32)
        if len(boxes) > 0:
            boxes   = np.array(boxes,   dtype=np.float32)
            cls_ids = np.array(cls_ids, dtype=np.float32)
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / s
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / s
            bw = (boxes[:, 2] - boxes[:, 0]) / s
            bh = (boxes[:, 3] - boxes[:, 1]) / s
            valid = (bw > 0.001) & (bh > 0.001)
            cx, cy, bw, bh, cls_ids = cx[valid], cy[valid], bw[valid], bh[valid], cls_ids[valid]
            if len(cx):
                target = torch.zeros((len(cx), 5), dtype=torch.float32)
                target[:, 0] = torch.tensor(cls_ids)
                target[:, 1] = torch.tensor(cx)
                target[:, 2] = torch.tensor(cy)
                target[:, 3] = torch.tensor(bw)
                target[:, 4] = torch.tensor(bh)
        meta = torch.tensor([w0, h0, pad_l, pad_t, scale], dtype=torch.float32)
        return img, target, str(ip), meta

    def _load_ann(self, stem, w, h):
        boxes, cls_ids = [], []
        ann_file = self.ann_dir / (stem + ".txt")
        if not ann_file.exists():
            return boxes, cls_ids
        try:
            with open(ann_file) as fp:
                for line in fp:
                    p = line.strip().split(",")
                    if len(p) < 6:
                        continue
                    x, y, bw, bh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                    score, cat = int(p[4]), int(p[5])
                    if cat == 0 or score == 0 or cat == 11:
                        continue
                    x1 = max(0., x);    y1 = max(0., y)
                    x2 = min(float(w), x + bw); y2 = min(float(h), y + bh)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2]); cls_ids.append(cat - 1)
        except Exception:
            pass
        return boxes, cls_ids

    def _augment(self, img, boxes, cls_ids, w0, h0):
        # Horizontal flip
        if random.random() > 0.5:
            img   = cv2.flip(img, 1)
            boxes = [[w0 - b[2], b[1], w0 - b[0], b[3]] for b in boxes]
        return img, boxes, cls_ids

    def _letterbox(self, img, boxes):
        h, w = img.shape[:2]; s = self.img_size
        scale = s / max(h, w); nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh))
        pl = (s - nw) // 2; pt = (s - nh) // 2
        pr = s - nw - pl;   pb = s - nh - pt
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        new_boxes = [[b[0]*scale+pl, b[1]*scale+pt,
                      b[2]*scale+pl, b[3]*scale+pt] for b in boxes]
        return img, new_boxes, pl, pt, scale


def collate_fn(batch):
    imgs, targets, paths, metas = zip(*batch)
    imgs  = torch.stack(imgs, 0)
    metas = torch.stack(metas, 0)
    bt = []
    for i, t in enumerate(targets):
        if len(t):
            bi = torch.full((len(t), 1), float(i))
            bt.append(torch.cat([bi, t], dim=1))
    bt = torch.cat(bt, 0) if bt else torch.zeros((0, 6))
    return imgs, bt, list(paths), metas


def build_dataloaders():
    tds = VisDroneDataset(
        os.path.join(TRAIN_SPLIT, "images"),
        os.path.join(TRAIN_SPLIT, "annotations"),
        IMG_SIZE, augment=True)
    vds = VisDroneDataset(
        os.path.join(VAL_SPLIT, "images"),
        os.path.join(VAL_SPLIT, "annotations"),
        IMG_SIZE, augment=False)
    if len(tds) == 0 or len(vds) == 0:
        raise RuntimeError("Dataset empty — check paths.")
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available(), drop_last=True,
                    persistent_workers=(NUM_WORKERS > 0))
    vl = DataLoader(vds, batch_size=1, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available())
    return tl, vl, tds, vds


# ================================================================
# 4. ARCHITECTURE
# ================================================================

class ASRFR(nn.Module):
    """
    Adaptive Sub-Resolution Feature Refinement.
    PixelUnshuffle(2) → 3×Conv → PixelShuffle(2), residual with
    learnable alpha clamped to [0,1].
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_shuffle   = nn.PixelShuffle(2)
        self.sr_net = nn.Sequential(
            nn.Conv2d(12, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 12, 3, 1, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        packed = self.pixel_unshuffle(x)
        sr     = self.pixel_shuffle(self.sr_net(packed))
        return x + self.alpha.clamp(0., 1.) * sr


class SpatialSSMBlock(nn.Module):
    """
    Mamba-style spatial selective state-space block.
    7×7 DWConv + BN → PW1 → GELU → PW2, residual scaled by
    per-channel learnable gamma (initialised to 0 → identity at init).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw    = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        self.norm  = nn.BatchNorm2d(dim)
        self.pw1   = nn.Conv2d(dim, dim * 2, 1)
        self.act   = nn.GELU()
        self.pw2   = nn.Conv2d(dim * 2, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        r = x
        x = self.norm(self.dw(x))
        x = self.pw2(self.act(self.pw1(x)))
        return r + self.gamma * x


class PassThrough(nn.Module):
    """
    Drop-in replacement for SpatialSSMBlock when SSM is disabled.
    Used for v3 / v5 / v7: same interface, zero extra parameters.
    A 1×1 Conv pass-through ensures channel dims are always correct
    even if a future refactor changes channel counts.
    """
    def __init__(self, dim: int):
        super().__init__()
        # nn.Identity() works here because channel dims match, but
        # we explicitly register it for cleaner parameter summaries.
        self._identity = nn.Identity()

    def forward(self, x):
        return self._identity(x)


class HiSMDNetVariant(nn.Module):
    """
    Single model class that covers all 7 ablation variants by
    toggling three boolean flags:
        use_asrfr   – prepend ASRFR sub-pixel refinement
        use_ssm     – insert SpatialSSMBlock after stage2/stage3
        use_stride2 – add stride-2 hi-res detection head

    Variant map:
        v1  True  True  True
        v2  False True  True
        v3  True  False True
        v4  True  True  False
        v5  False False True
        v6  False True  False
        v7  True  False False

    BUG FIX (vs. original train.py):
        In the original, pb (decoded predicted boxes) was computed
        inside `torch.no_grad()` for the assignment step, then the
        SAME detached pb tensor was reused in `ciou_loss()`.  This
        broke the gradient path from IoU loss → reg_heads.

        Fix: the assignment uses detached pb (correct — no grad needed
        for target assignment), but ciou_loss() is called with pb
        re-decoded OUTSIDE no_grad() so gradients flow correctly.
    """

    def __init__(self, variant_id: int, neck_ch: int = 128,
                 asrfr_ch: int = 32, nc: int = 10):
        super().__init__()
        cfg = VARIANT_CONFIGS[variant_id]
        self.variant_id   = variant_id
        self.variant_name = cfg["name"]
        self.nc           = nc
        self.use_asrfr    = cfg["use_asrfr"]
        self.use_ssm      = cfg["use_ssm"]
        self.use_stride2  = cfg["use_stride2"]

        # Detection strides depend on whether stride-2 head is active
        self.strides = [2, 4, 8, 16] if self.use_stride2 else [4, 8, 16]
        self.n_heads = len(self.strides)

        # ── Optional ASRFR preprocessor ───────────────────────────────
        if self.use_asrfr:
            self.asrfr = ASRFR(asrfr_ch)

        # ── Backbone: ResNet-50 (pretrained weights on ImageNet) ──────
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stage1 = base.layer1   # output: stride-4,  256ch

        # v1/v2/v4/v6 use SSM; v3/v5/v7 use PassThrough (identity)
        SSM = SpatialSSMBlock if self.use_ssm else PassThrough
        self.stage2 = nn.Sequential(base.layer2, SSM(512))   # stride-8,  512ch
        self.stage3 = nn.Sequential(base.layer3, SSM(1024))  # stride-16, 1024ch

        # ── FPN lateral + fuse convs ──────────────────────────────────
        self.lat3 = nn.Conv2d(1024, neck_ch, 1)
        self.lat2 = nn.Conv2d(512,  neck_ch, 1)
        self.lat1 = nn.Conv2d(256,  neck_ch, 1)

        self.fuse3 = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True))
        self.fuse2 = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True))
        self.fuse1 = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True))

        # ── Optional Stride-2 hi-res head ─────────────────────────────
        # Sub-pixel upsampling: Conv → PixelShuffle(2) → BN+SiLU → Conv
        if self.use_stride2:
            self.hires_head = nn.Sequential(
                nn.Conv2d(neck_ch, neck_ch * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(neck_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(neck_ch, neck_ch, 3, 1, 1),
                nn.BatchNorm2d(neck_ch),
                nn.SiLU(inplace=True),
            )

        # ── Decoupled cls + reg detection heads ───────────────────────
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.SiLU(inplace=True),
                nn.Conv2d(neck_ch, nc, 1)
            ) for _ in range(self.n_heads)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.SiLU(inplace=True),
                nn.Conv2d(neck_ch, 4, 1)
            ) for _ in range(self.n_heads)
        ])

        # ── Weight init ───────────────────────────────────────────────
        prior = 0.01
        for h in self.cls_heads:
            nn.init.constant_(h[-1].bias, -math.log((1 - prior) / prior))
        for h in self.reg_heads:
            nn.init.constant_(h[-1].bias, 0.5)

        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"\n  HiSMDNetVariant [v{variant_id}:{self.variant_name}]"
              f" | Params: {total_p:.2f}M"
              f" | ASRFR={self.use_asrfr} SSM={self.use_ssm}"
              f" Stride2={self.use_stride2} | Strides={self.strides}")

    def forward(self, x):
        # 1. Optional ASRFR sub-pixel refinement
        if self.use_asrfr:
            x = self.asrfr(x)

        # 2. Backbone
        s  = self.stem(x)
        c2 = self.stage1(s)    # stride-4
        c3 = self.stage2(c2)   # stride-8
        c4 = self.stage3(c3)   # stride-16

        # 3. FPN (top-down)
        p4 = self.lat3(c4)
        p3 = self.fuse3(self.lat2(c3) + F.interpolate(p4, scale_factor=2, mode="nearest"))
        p2 = self.fuse2(self.lat1(c2) + F.interpolate(p3, scale_factor=2, mode="nearest"))
        p1_base = self.fuse1(p2)   # stride-4 feature map

        # 4. Feature pyramid for detection
        if self.use_stride2:
            p1_hires = self.hires_head(p1_base)   # stride-2 (320×320 for 640 input)
            feats = [p1_hires, p1_base, p3, p4]
        else:
            feats = [p1_base, p3, p4]

        # 5. Detection heads — Softplus + clamp for smooth, bounded LTRB offsets
        cls_outs = [self.cls_heads[i](f) for i, f in enumerate(feats)]
        reg_outs = [F.softplus(self.reg_heads[i](f)).clamp(max=8.0)
                    for i, f in enumerate(feats)]

        return cls_outs, reg_outs


# ================================================================
# 5. ANCHOR / LOSS
# ================================================================

def make_anchors(cls_outs, strides, device=None):
    pts, strs = [], []
    if device is None:
        device = cls_outs[0].device
    for c, s in zip(cls_outs, strides):
        H, W = c.shape[-2:]
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing="ij")
        pts.append(torch.stack([(gx.flatten() + 0.5) * s,
                                (gy.flatten() + 0.5) * s], 1))
        strs.append(torch.full((H * W,), s, device=device, dtype=torch.float32))
    return torch.cat(pts), torch.cat(strs)


def decode_boxes(reg, pts, strs):
    """LTRB → xyxy. reg shape (A,4), pts/strs shape (A,)."""
    off = reg * strs.unsqueeze(1)
    return torch.stack([pts[:, 0] - off[:, 0], pts[:, 1] - off[:, 1],
                        pts[:, 0] + off[:, 2], pts[:, 1] + off[:, 3]], 1)


def ciou_loss(p, g, eps: float = 1e-7):
    """Complete IoU loss (returns per-box loss)."""
    ix1 = torch.max(p[:, 0], g[:, 0]); iy1 = torch.max(p[:, 1], g[:, 1])
    ix2 = torch.min(p[:, 2], g[:, 2]); iy2 = torch.min(p[:, 3], g[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    ap = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    ag = (g[:, 2] - g[:, 0]).clamp(0) * (g[:, 3] - g[:, 1]).clamp(0)
    union = ap + ag - inter + eps; iou = (inter / union).clamp(0, 1)
    ex1 = torch.min(p[:, 0], g[:, 0]); ey1 = torch.min(p[:, 1], g[:, 1])
    ex2 = torch.max(p[:, 2], g[:, 2]); ey2 = torch.max(p[:, 3], g[:, 3])
    enc_diag = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + eps
    cx_p = (p[:, 0] + p[:, 2]) * 0.5; cy_p = (p[:, 1] + p[:, 3]) * 0.5
    cx_g = (g[:, 0] + g[:, 2]) * 0.5; cy_g = (g[:, 1] + g[:, 3]) * 0.5
    ctr  = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2
    ciou = iou - (ctr / enc_diag).clamp(0, 1)
    return 1. - ciou.clamp(-1., 1.)


def assign_and_loss(cls_outs, reg_outs, targets, strides, img_size, nc,
                    wcls, wbox, wl1, device):
    """
    Task-Aligned Assignment + CIoU + smooth-L1 regression loss.

    BUG FIX:  pb_for_assign is computed under no_grad() (correct — we
    only need detached predictions for the assignment score).
    pb_for_loss is decoded OUTSIDE no_grad() so that ciou_loss can
    back-prop through reg_outs → reg_heads.
    """
    B  = cls_outs[0].shape[0]
    pts, strs = make_anchors(cls_outs, strides, device=device)
    A = pts.shape[0]

    # Flatten predictions
    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, nc) for c in cls_outs], 1)
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4)  for r in reg_outs], 1)

    # ── For ASSIGNMENT only — detached, no gradient ──────────────────
    with torch.no_grad():
        pb_assign = torch.stack([
            decode_boxes(reg_f[b].detach(), pts, strs).clamp(0, img_size)
            for b in range(B)])

    # ── GT boxes in pixel coords ──────────────────────────────────────
    N = len(targets)
    gt_px = torch.zeros(N, 4, device=device)
    if N > 0:
        gt_px[:, 0] = (targets[:, 2] - targets[:, 4] / 2) * img_size
        gt_px[:, 1] = (targets[:, 3] - targets[:, 5] / 2) * img_size
        gt_px[:, 2] = (targets[:, 2] + targets[:, 4] / 2) * img_size
        gt_px[:, 3] = (targets[:, 3] + targets[:, 5] / 2) * img_size

    cls_tgt = torch.zeros(B, A, nc,  device=device)
    reg_tgt = torch.zeros(B, A, 4,   device=device)
    box_tgt = torch.zeros(B, A, 4,   device=device)
    wts     = torch.zeros(B, A,      device=device)

    with torch.no_grad():
        for b in range(B):
            if N == 0:
                continue
            bm = targets[:, 0] == b
            if not bm.any():
                continue
            gtb = gt_px[bm]; clb = targets[bm, 1].long(); G = len(gtb)
            xa = pts[:, 0].unsqueeze(0); ya = pts[:, 1].unsqueeze(0)
            l_star = (xa - gtb[:, 0:1]) / strs.unsqueeze(0)
            t_star = (ya - gtb[:, 1:2]) / strs.unsqueeze(0)
            r_star = (gtb[:, 2:3] - xa) / strs.unsqueeze(0)
            b_star = (gtb[:, 3:4] - ya) / strs.unsqueeze(0)
            offsets_star = torch.stack([l_star, t_star, r_star, b_star], dim=-1)
            radius = (strs * 1.5).unsqueeze(0)
            is_cand = ((xa >= gtb[:, 0:1] - radius) & (xa <= gtb[:, 2:3] + radius) &
                       (ya >= gtb[:, 1:2] - radius) & (ya <= gtb[:, 3:4] + radius) &
                       (l_star >= 0) & (t_star >= 0) & (r_star >= 0) & (b_star >= 0)).float()
            chunk = 8192; iou_chunks = []
            for ci in range(0, A, chunk):
                iou_chunks.append(
                    box_iou(gtb, pb_assign[b][ci:ci + chunk]).clamp(0, 1))
            iou_ga  = torch.cat(iou_chunks, dim=1)
            csc_ga  = torch.sigmoid(cls_f[b].detach())[:, clb].T.clamp(1e-6, 1.)
            ta_sc   = (iou_ga ** 0.5) * (csc_ga ** 0.5) * is_cand
            k = min(10, A)
            topk_v, topk_i = ta_sc.topk(k, dim=1)
            vf = topk_v.flatten(); vi = topk_i.flatten()
            cf = clb.unsqueeze(1).expand(-1, k).flatten()
            bf = gtb.unsqueeze(1).expand(-1, k, 4).reshape(-1, 4)
            rf = offsets_star.view(G, A, 4).gather(
                1, topk_i.unsqueeze(-1).expand(-1, -1, 4)).reshape(-1, 4)
            ok = vf > 0.005
            if not ok.any():
                continue
            vf, vi, cf, bf, rf = vf[ok], vi[ok], cf[ok], bf[ok], rf[ok]
            cur = wts[b]; better = vf > cur[vi]
            vi_b, vf_b, cf_b, bf_b, rf_b = vi[better], vf[better], cf[better], bf[better], rf[better]
            if len(vi_b) == 0:
                continue
            cur[vi_b] = vf_b
            cls_tgt[b, vi_b, :] = 0.; cls_tgt[b, vi_b, cf_b] = vf_b
            reg_tgt[b, vi_b] = rf_b; box_tgt[b, vi_b] = bf_b

    pos  = wts > 0
    npos = pos.sum().clamp(min=1)

    # Classification loss (all anchors)
    cls_loss = F.binary_cross_entropy_with_logits(
        cls_f.reshape(-1, nc), cls_tgt.reshape(-1, nc), reduction="sum") / npos

    box_loss = l1_loss = torch.tensor(0., device=device, requires_grad=False)

    if pos.any():
        # ── FIXED: re-decode OUTSIDE no_grad so gradients flow ──────
        pb_for_loss = decode_boxes(
            reg_f[pos], pts.unsqueeze(0).expand(B, -1, -1)[pos],
            strs.unsqueeze(0).expand(B, -1)[pos]
        ).clamp(0, img_size)
        box_loss = ciou_loss(pb_for_loss, box_tgt[pos]).sum() / npos

        if wl1 > 0:
            l1_loss = F.smooth_l1_loss(
                reg_f[pos], reg_tgt[pos].clamp(0, 8.),
                reduction="sum") / npos * wl1

    total_loss = wcls * cls_loss + wbox * box_loss + l1_loss
    return total_loss, cls_loss.detach(), box_loss.detach()


# ================================================================
# 6. POSTPROCESSING & COCO EVAL
# ================================================================

def postprocess(cls_outs, reg_outs, strides, img_size, nc,
                conf: float = 0.05, max_det: int = 300):
    B = cls_outs[0].shape[0]
    pts, strs = make_anchors(cls_outs, strides, device=cls_outs[0].device)
    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, nc).sigmoid()
                        for c in cls_outs], 1)
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4)
                        for r in reg_outs], 1)
    results = []
    for b in range(B):
        pb = decode_boxes(reg_f[b], pts, strs).clamp(0, img_size)
        sc, lb = cls_f[b].max(-1)
        mask = sc > conf
        pb, sc, lb = pb[mask], sc[mask], lb[mask]
        if len(pb) == 0:
            results.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                            "labels": torch.zeros(0, dtype=torch.long)})
            continue
        fb, fs, fl = [], [], []
        for c in range(nc):
            m = lb == c
            if not m.any():
                continue
            keep = nms(pb[m], sc[m], NMS_IOU)
            fb.append(pb[m][keep]); fs.append(sc[m][keep])
            fl.append(torch.full((len(keep),), c, dtype=torch.long, device=pb.device))
        if fb:
            fb, fs, fl = torch.cat(fb), torch.cat(fs), torch.cat(fl)
            if len(fs) > max_det:
                tk = fs.topk(max_det).indices; fb, fs, fl = fb[tk], fs[tk], fl[tk]
            results.append({"boxes": fb.cpu(), "scores": fs.cpu(), "labels": fl.cpu()})
        else:
            results.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                            "labels": torch.zeros(0, dtype=torch.long)})
    return results


def build_coco_gt(vds: VisDroneDataset) -> dict:
    gt = {"images": [], "annotations": [],
          "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]}
    aid = 0
    for iid, ip in enumerate(vds.img_paths):
        im = cv2.imread(str(ip))
        h0, w0 = im.shape[:2] if im is not None else (IMG_SIZE, IMG_SIZE)
        gt["images"].append({"id": iid, "width": w0, "height": h0, "file_name": ip.name})
        ap = vds.ann_dir / (ip.stem + ".txt")
        if not ap.exists():
            continue
        for line in ap.read_text().splitlines():
            p = line.split(",")
            if len(p) < 6:
                continue
            x, y, bw, bh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            cat, sc = int(p[5]), int(p[4])
            if cat == 0 or sc == 0 or cat == 11:
                continue
            gt["annotations"].append({
                "id": aid, "image_id": iid, "category_id": cat - 1,
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    return gt


@torch.no_grad()
def evaluate_coco(model, val_dl, gt_dict, device, conf: float = 0.05):
    import tempfile
    model.eval()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); gf = f.name
    coco_gt = COCO(gf)
    preds = []; iid = 0
    for imgs, _, paths, metas in tqdm(val_dl, desc="Eval", leave=False, file=sys.stdout):
        imgs = imgs.to(device)
        with autocast(enabled=USE_AMP):
            co, ro = model(imgs)
        dets = postprocess(co, ro, model.strides, IMG_SIZE, NUM_CLASSES, conf)
        for b, det in enumerate(dets):
            w0, h0, pl, pt, sc = metas[b].tolist()
            for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                x1, y1, x2, y2 = box.tolist()
                x1o = max(0., (x1 - pl) / sc); y1o = max(0., (y1 - pt) / sc)
                x2o = min(w0,  (x2 - pl) / sc); y2o = min(h0,  (y2 - pt) / sc)
                wo = max(0., x2o - x1o); ho = max(0., y2o - y1o)
                if wo > 0 and ho > 0:
                    preds.append({"image_id": iid + b, "category_id": int(label),
                                  "bbox": [x1o, y1o, wo, ho], "score": float(score)})
        iid += len(dets)
    empty = {"AP": 0., "AP50": 0., "APS": 0., "APM": 0.}
    if not preds:
        return empty
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f); df = f.name
    try:
        ev = COCOeval(coco_gt, coco_gt.loadRes(df), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return {"AP": ev.stats[0], "AP50": ev.stats[1], "APS": ev.stats[3], "APM": ev.stats[4]}
    except Exception as e:
        print(f"  [!] COCO eval error: {e}")
        return empty


@torch.no_grad()
def benchmark_inference_speed(model, device, img_size: int = 640,
                              num_warmup: int = 15, num_runs: int = 50) -> dict:
    """
    Measures real inference latency (ms per image) and throughput (FPS).
    Uses synthetic batch of size 1 with CUDA event/synchronize timing.
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    # Warmup
    for _ in range(num_warmup):
        with autocast(enabled=USE_AMP):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        with autocast(enabled=USE_AMP):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    avg_latency_ms = (elapsed / num_runs) * 1000.0
    fps = num_runs / elapsed
    return {"latency_ms": round(avg_latency_ms, 2), "fps": round(fps, 1)}


# ================================================================
# 7. VISUALISATION (optional — skip with --no_vis)
# ================================================================

def save_vis(model, val_dl, strides, save_path, epoch, max_imgs=4):
    model.eval()
    fig, axes = plt.subplots(1, max_imgs, figsize=(max_imgs * 5, 5))
    shown = 0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        for imgs, _, paths, _ in val_dl:
            if shown >= max_imgs:
                break
            imgs_d = imgs.to(DEVICE)
            with autocast(enabled=USE_AMP):
                co, ro = model(imgs_d)
            dets = postprocess(co, ro, strides, IMG_SIZE, NUM_CLASSES, VIS_CONF, MAX_DET)
            for b in range(len(imgs)):
                if shown >= max_imgs:
                    break
                img_np = imgs[b].cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np * std + mean, 0, 1)
                ax = axes[shown] if max_imgs > 1 else axes
                ax.imshow(img_np)
                det = dets[b]
                for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    color = [c / 255 for c in COLORS[int(label) % len(COLORS)]]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=1, edgecolor=color, facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 2, f"{CLASS_NAMES[int(label)]} {score:.2f}",
                            color=color, fontsize=6)
                ax.set_title(f"Ep{epoch:03d}", fontsize=8)
                ax.axis("off")
                shown += 1
    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# 8. TRAINING LOOP (single variant)
# ================================================================

def train_one_variant(variant_id: int):
    cfg  = VARIANT_CONFIGS[variant_id]
    name = cfg["name"]
    save_dir = ABLATION_DIR / f"v{variant_id}_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = str(save_dir / f"best_v{variant_id}.pt")
    last_ckpt = str(save_dir / f"last_v{variant_id}.pt")
    log_csv   = str(save_dir / "training_log.csv")
    cfg_json  = str(save_dir / "variant_config.json")

    print("\n" + "=" * 70)
    print(f"  VARIANT v{variant_id}: {name.upper()}")
    print(f"  ASRFR={cfg['use_asrfr']}  SSM={cfg['use_ssm']}  Stride2={cfg['use_stride2']}")
    print(f"  Epochs={EPOCHS}  Seed={SEED}  Batch={BATCH_SIZE}  LR={args.lr}")
    print(f"  Output: {save_dir}")
    print("=" * 70)

    # Save variant config for reproducibility
    with open(cfg_json, "w") as f:
        json.dump({"variant_id": variant_id, "name": name, **cfg,
                   "epochs": EPOCHS, "seed": SEED, "batch_size": BATCH_SIZE,
                   "lr": args.lr, "img_size": IMG_SIZE, "nc": NUM_CLASSES}, f, indent=2)

    set_seed(SEED)
    tl, vl, tds, vds = build_dataloaders()
    gt_dict = build_coco_gt(vds)

    model  = HiSMDNetVariant(variant_id, NECK_CH, ASRFR_CH, NUM_CLASSES).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=MIN_LR)
    scaler = GradScaler(enabled=USE_AMP)

    # ── Write CSV header (only if fresh run) ─────────────────────────
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "loss", "cls_loss", "box_loss",
                                    "AP", "AP50", "APS", "APM"])

    # ── Auto-resume ───────────────────────────────────────────────────
    start_epoch = 1; best_ap50 = 0.; best_ap = 0.
    for ckpt_path in [last_ckpt, best_ckpt]:
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:  opt.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:  sched.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and USE_AMP and ckpt["scaler"]:
                    scaler.load_state_dict(ckpt["scaler"])
                start_epoch = ckpt.get("epoch", 1) + 1
                best_ap50   = ckpt.get("best_ap50", 0.)
                best_ap     = ckpt.get("best_ap", 0.)
                print(f"  [+] Resumed from {ckpt_path} at epoch {start_epoch - 1}")
                break
            except Exception as e:
                print(f"  [!] Resume failed ({e}) — starting fresh")

    if start_epoch > EPOCHS:
        print(f"  [=] v{variant_id} already complete ({EPOCHS} epochs). Skipping.")
        return

    # ── Main training loop ────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        tot = cls_s = box_s = 0.; nb = 0
        pbar = tqdm(tl, desc=f"[v{variant_id}:{name}] Ep {epoch:3d}/{EPOCHS}",
                    leave=False, file=sys.stdout, ncols=110)
        for imgs, tgts, _, _ in pbar:
            imgs = imgs.to(DEVICE); tgts = tgts.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                co, ro = model(imgs)
                loss, cl, bl = assign_and_loss(
                    co, ro, tgts, model.strides, IMG_SIZE,
                    NUM_CLASSES, W_CLS, W_BOX, W_L1, DEVICE)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [!] NaN/Inf loss at epoch {epoch} — skipping batch")
                continue
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            tot += loss.item(); cls_s += cl.item(); box_s += bl.item(); nb += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}",
                              "cls": f"{cl.item():.3f}",
                              "box": f"{bl.item():.3f}"})

        sched.step()
        lr  = sched.get_last_lr()[0]
        avg = {"total": tot / max(nb, 1), "cls": cls_s / max(nb, 1), "box": box_s / max(nb, 1)}
        print(f"\n  [v{variant_id}] Ep {epoch:3d}/{EPOCHS} | LR={lr:.2e} | "
              f"Loss={avg['total']:.4f} (cls={avg['cls']:.4f} box={avg['box']:.4f})")

        m = None
        if epoch % EVAL_FREQ == 0 or epoch == EPOCHS:
            m = evaluate_coco(model, vl, gt_dict, DEVICE, CONF_THRESH)
            print(f"    >>> AP={m['AP']*100:.2f}% | AP50={m['AP50']*100:.2f}% | "
                  f"APS={m['APS']*100:.2f}% | APM={m['APM']*100:.2f}%")

            # Save vis
            if not args.no_vis:
                vis_path = str(save_dir / f"vis_epoch_{epoch:03d}.png")
                try:
                    save_vis(model, vl, model.strides, vis_path, epoch)
                except Exception as e:
                    print(f"    [!] Vis error: {e}")

            if m["AP50"] > best_ap50 or m["AP"] > best_ap:
                best_ap50 = max(best_ap50, m["AP50"])
                best_ap   = max(best_ap,   m["AP"])
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict() if USE_AMP else None,
                    "best_ap50": best_ap50, "best_ap": best_ap,
                    "variant_id": variant_id, "variant_name": name,
                }, best_ckpt)
                print(f"    *** NEW BEST: AP50={best_ap50*100:.2f}% "
                      f"mAP={best_ap*100:.2f}% → {best_ckpt}")

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if USE_AMP else None,
            "best_ap50": best_ap50, "best_ap": best_ap,
            "variant_id": variant_id, "variant_name": name,
        }, last_ckpt)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.6e}",
                f"{avg['total']:.4f}", f"{avg['cls']:.4f}", f"{avg['box']:.4f}",
                f"{m['AP']*100:.2f}"   if m else "",
                f"{m['AP50']*100:.2f}" if m else "",
                f"{m['APS']*100:.2f}"  if m else "",
                f"{m['APM']*100:.2f}"  if m else "",
            ])

    # ── Final speed benchmark ─────────────────────────────────────────
    print(f"\n  Benchmarking inference speed for v{variant_id}:{name} ...")
    if os.path.exists(best_ckpt):
        try:
            b_data = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(b_data["model"])
        except Exception:
            pass
    speed = benchmark_inference_speed(model, DEVICE, IMG_SIZE)
    speed_file = save_dir / "speed_benchmark.json"
    with open(speed_file, "w") as f:
        json.dump(speed, f, indent=2)
    print(f"  Speed: {speed['latency_ms']:.2f} ms/img ({speed['fps']:.1f} FPS)")

    # Final summary
    print("\n" + "=" * 70)
    print(f"  [v{variant_id}:{name}] DONE.")
    print(f"  Best AP50: {best_ap50*100:.2f}%  mAP: {best_ap*100:.2f}%")
    print(f"  Speed: {speed['latency_ms']:.2f} ms ({speed['fps']:.1f} FPS)")
    print(f"  Outputs:  {save_dir}")
    print("=" * 70)


# ================================================================
# 9. GENERATE COMBINED CSV COMPARISON TABLE
# ================================================================

def generate_comparison_csv():
    """
    Reads the best_vN.pt checkpoints (or training_log.csv) for every
    variant and emits ablation_output/ablation_comparison.csv.
    Only includes variants where results actually exist.
    """
    rows = []
    for vid, cfg in VARIANT_CONFIGS.items():
        name    = cfg["name"]
        vdir    = ABLATION_DIR / f"v{vid}_{name}"
        log_csv = vdir / "training_log.csv"
        if not log_csv.exists():
            continue
        # Load best result from log
        best_ap50 = best_ap = best_aps = best_apm = 0.
        try:
            with open(log_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ap50 = float(row.get("AP50", 0) or 0)
                        ap   = float(row.get("AP",   0) or 0)
                        aps  = float(row.get("APS",  0) or 0)
                        apm  = float(row.get("APM",  0) or 0)
                        if ap50 > best_ap50:
                            best_ap50 = ap50; best_ap = ap
                            best_aps = aps; best_apm = apm
                    except ValueError:
                        pass
        except Exception as e:
            print(f"  [!] Could not read {log_csv}: {e}")
            continue

        # Count parameters from checkpoint
        params_m = "N/A"
        ckpt_path = vdir / f"best_v{vid}.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                total = sum(p.numel() for p in
                            HiSMDNetVariant(vid, NECK_CH, ASRFR_CH, NUM_CLASSES).parameters()
                            if p.requires_grad)
                params_m = f"{total/1e6:.2f}M"
            except Exception:
                pass

        # Load speed benchmark if available
        latency_str = "N/A"
        fps_str = "N/A"
        speed_json = vdir / "speed_benchmark.json"
        if speed_json.exists():
            try:
                with open(speed_json) as f:
                    s_data = json.load(f)
                    latency_str = f"{s_data.get('latency_ms', 0):.2f}"
                    fps_str     = f"{s_data.get('fps', 0):.1f}"
            except Exception:
                pass

        rows.append({
            "Variant":      f"v{vid}",
            "Name":         name,
            "ASRFR":        str(cfg["use_asrfr"]),
            "SpatialSSM":   str(cfg["use_ssm"]),
            "Stride2":      str(cfg["use_stride2"]),
            "AP50 (%)":     f"{best_ap50:.2f}",
            "mAP (%)":      f"{best_ap:.2f}",
            "APS (%)":      f"{best_aps:.2f}",
            "APM (%)":      f"{best_apm:.2f}",
            "Params":       params_m,
            "Latency (ms)": latency_str,
            "FPS":          fps_str,
        })

    if not rows:
        print("\n[!] No variant results found yet — comparison CSV not written.")
        return

    out_csv = str(ABLATION_DIR / "ablation_comparison.csv")
    fieldnames = ["Variant","Name","ASRFR","SpatialSSM","Stride2",
                  "AP50 (%)","mAP (%)","APS (%)","APM (%)",
                  "Params","Latency (ms)","FPS"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    print(f"\n{'='*75}")
    print(f"  Ablation comparison table → {out_csv}")
    print(f"{'='*75}")
    # Pretty-print to console
    header = f"{'Var':<5} {'Name':<22} {'ASRFR':<7} {'SSM':<6} {'S2':<6} " \
             f"{'AP50':>7} {'mAP':>7} {'APS':>7} {'APM':>7} {'Params':>9} {'ms':>7} {'FPS':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"  {r['Variant']:<4} {r['Name']:<22} {r['ASRFR']:<7} "
              f"{r['SpatialSSM']:<6} {r['Stride2']:<6} "
              f"{r['AP50 (%)']:>7} {r['mAP (%)']:>7} "
              f"{r['APS (%)']:>7} {r['APM (%)']:>7} "
              f"{r['Params']:>9} {r['Latency (ms)']:>7} {r['FPS']:>7}")
    print("=" * 75)


# ================================================================
# 10. ENTRY POINT
# ================================================================

def main():
    print("\n" + "=" * 70)
    print("  HiSMD-Net Ablation Study  |  train_v2.py")
    print(f"  Variants to run : {VARIANTS_TO_RUN}")
    print(f"  Epochs: {EPOCHS}  |  Seed: {SEED}  |  Batch: {BATCH_SIZE}"
          f"  |  Device: {DEVICE}")
    print(f"  AMP: {USE_AMP}  |  GPU VRAM: {_vram:.1f} GB")
    print(f"  Train split: {TRAIN_SPLIT}")
    print(f"  Val   split: {VAL_SPLIT}")
    print(f"  Output root: {ABLATION_DIR}")
    print("=" * 70)

    for vid in VARIANTS_TO_RUN:
        train_one_variant(vid)

    # Generate combined comparison table
    generate_comparison_csv()

    print("\n[✓] All requested variants complete.\n")


if __name__ == "__main__":
    main()
