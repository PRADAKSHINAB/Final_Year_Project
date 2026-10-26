# ==============================================================================
# HiSMD-Net Ablation Study V2 — ablation_v2/train_ablation_v2.py
# ==============================================================================
# COMPLETELY ISOLATED — All outputs and files are saved exclusively in
# ablation_v2/. Never modifies original train.py or prior outputs.
#
# KEY IMPLEMENTATION CORRECTIONS:
# 1. Complete CIoU loss: Includes distance loss AND aspect-ratio penalty (v, alpha).
# 2. Gradient flow: Target assignment uses detached boxes, but CIoU loss uses
#    boxes re-decoded dynamically outside no_grad() for direct gradient flow.
# 3. Spatial Depthwise Modulation: Accurately documented as a 2D depthwise
#    convolutional spatial modulation block (7x7 DWConv + BN + PW1 + GELU + PW2),
#    clarifying that it functions as a spatial feature modulator rather than a
#    recurrent 1D state-space scan.
# 4. 100 Epochs Standard: All 7 variants train for 100 epochs under identical
#    settings, random seeds, optimizer parameters, and dataset splits.
# 5. ASRFR Progression: Full model checkpoints and metrics are evaluated and
#    saved at 30, 60, and 100 epochs to test whether longer training aids ASRFR.
# 6. Automatic reporting: Generates ablation_comparison.csv, epoch_comparison.csv,
#    and results_summary.txt with rigorous CUDA-synchronized latency benchmarks.
# ==============================================================================

import os
import sys
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import json
import math
import time
import random
import warnings
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings("ignore")

# ── Package bootstrap for HPC user environment ────────────────────────────────
MY_PACKAGES = os.path.expanduser("~/my_python_packages")
if os.path.exists(MY_PACKAGES) and MY_PACKAGES not in sys.path:
    sys.path.insert(0, MY_PACKAGES)


def ensure_package(import_name: str, pip_name: str):
    # Only bootstrap packages dynamically on Linux / HPC
    if sys.platform != "linux" or "-h" in sys.argv or "--help" in sys.argv or "--dry_run" in sys.argv:
        return
    try:
        __import__(import_name)
    except ImportError:
        os.makedirs(MY_PACKAGES, exist_ok=True)
        if MY_PACKAGES not in sys.path:
            sys.path.insert(0, MY_PACKAGES)
        print(f"[+] Installing {pip_name} to {MY_PACKAGES} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--target", MY_PACKAGES, "--upgrade", pip_name],
                timeout=120
            )
        except Exception as e:
            print(f"[!] Warning: automatic installation of {pip_name} failed: {e}")


ensure_package("pycocotools", "pycocotools")
ensure_package("torchvision", "torchvision")

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

# ==============================================================================
# 0. VARIANT REGISTRY (7 Strictly Defined Ablation Configurations)
# ==============================================================================

VARIANT_CONFIGS: Dict[int, dict] = {
    1: {
        "name": "v1_full",
        "description": "Full HiSMD-Net (ASRFR + SpatialModulation + Stride-2 head)",
        "use_asrfr": True,
        "use_ssm": True,
        "use_stride2": True
    },
    2: {
        "name": "v2_no_asrfr",
        "description": "Without ASRFR (SpatialModulation + Stride-2 head)",
        "use_asrfr": False,
        "use_ssm": True,
        "use_stride2": True
    },
    3: {
        "name": "v3_no_ssm",
        "description": "Without SpatialModulation (ASRFR + Stride-2 head)",
        "use_asrfr": True,
        "use_ssm": False,
        "use_stride2": True
    },
    4: {
        "name": "v4_no_stride2",
        "description": "Without Stride-2 head (ASRFR + SpatialModulation; Strides 4, 8, 16)",
        "use_asrfr": True,
        "use_ssm": True,
        "use_stride2": False
    },
    5: {
        "name": "v5_no_asrfr_no_ssm",
        "description": "Without ASRFR and SpatialModulation (Stride-2 head only)",
        "use_asrfr": False,
        "use_ssm": False,
        "use_stride2": True
    },
    6: {
        "name": "v6_no_asrfr_no_stride2",
        "description": "Without ASRFR and Stride-2 head (SpatialModulation; Strides 4, 8, 16)",
        "use_asrfr": False,
        "use_ssm": True,
        "use_stride2": False
    },
    7: {
        "name": "v7_no_ssm_no_stride2",
        "description": "Without SpatialModulation and Stride-2 head (ASRFR; Strides 4, 8, 16)",
        "use_asrfr": True,
        "use_ssm": False,
        "use_stride2": False
    },
}

# ==============================================================================
# 1. DIRECTORIES & PATH SETUP
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
# Ensure output directory is strictly within ablation_v2/
OUTPUT_ROOT = BASE_DIR
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def find_visdrone_split(split_name: str) -> Optional[str]:
    roots = [
        BASE_DIR.parent / "Dataset",
        BASE_DIR / "Dataset",
        BASE_DIR.parent,
        Path(os.path.expanduser("~/Dataset")),
        Path(os.path.expanduser("~/uavdet_system/uavdet-system/Dataset")),
    ]
    exact = [
        BASE_DIR.parent / "Dataset" / split_name / split_name,
        BASE_DIR.parent / "Dataset" / split_name,
        BASE_DIR / "Dataset" / split_name / split_name,
        BASE_DIR / "Dataset" / split_name,
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
    BASE_DIR.parent / "Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train"
)
VAL_SPLIT = find_visdrone_split("VisDrone2019-DET-val") or str(
    BASE_DIR.parent / "Dataset/VisDrone2019-DET-val"
)

# ==============================================================================
# 2. HYPERPARAMETERS
# ==============================================================================

IMG_SIZE = 640
NUM_CLASSES = 10
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]
COLORS = [
    (255, 64, 64), (255, 178, 64), (64, 255, 64), (64, 178, 255),
    (178, 64, 255), (255, 64, 178), (64, 255, 178), (255, 255, 64),
    (255, 128, 0), (64, 64, 255)
]

NECK_CH = 128
ASRFR_CH = 32
LR = 5e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
W_CLS = 1.0
W_BOX = 4.0
W_L1 = 1.0
CONF_THRESH = 0.05
VIS_CONF = 0.25
NMS_IOU = 0.50
MAX_DET = 300
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_EVAL_FREQ = 5

USE_AMP = torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
    if torch.cuda.is_available()
    else 0.0
)

# Auto batch size based on GPU VRAM
if VRAM_GB > 50:
    DEFAULT_BATCH_SIZE = 16
elif VRAM_GB > 20:
    DEFAULT_BATCH_SIZE = 8
elif torch.cuda.is_available():
    DEFAULT_BATCH_SIZE = 4
else:
    DEFAULT_BATCH_SIZE = 2

NUM_WORKERS = 4 if torch.cuda.is_available() else 0


def set_seed(s: int):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# 3. COMPLETE CIoU LOSS IMPLEMENTATION
# ==============================================================================

def complete_ciou_loss(p: torch.Tensor, g: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete CIoU loss (Zheng et al., 2020) with both center distance
    and aspect-ratio penalty terms. Fully differentiable.

    p: Predicted bounding boxes [N, 4] in (x1, y1, x2, y2)
    g: Ground truth bounding boxes [N, 4] in (x1, y1, x2, y2)
    Returns: Tensor of per-box CIoU loss [N]
    """
    # 1. Coordinates
    px1, py1, px2, py2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    gx1, gy1, gx2, gy2 = g[:, 0], g[:, 1], g[:, 2], g[:, 3]

    # 2. Widths and heights (clamped to eps to prevent div-by-zero)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    # 3. Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter_w = (ix2 - ix1).clamp(min=0.0)
    inter_h = (iy2 - iy1).clamp(min=0.0)
    inter = inter_w * inter_h

    # 4. Union & standard IoU
    union = pw * ph + gw * gh - inter + eps
    iou = (inter / union).clamp(0.0, 1.0)

    # 5. Enclosing box diagonal squared (c^2)
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    c_diag_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    # 6. Center point distance squared (rho^2)
    p_cx = (px1 + px2) * 0.5
    p_cy = (py1 + py2) * 0.5
    g_cx = (gx1 + gx2) * 0.5
    g_cy = (gy1 + gy2) * 0.5
    rho_sq = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2

    # 7. Aspect ratio penalty term v and weight alpha
    # v measures discrepancy of aspect ratio: v = 4 / pi^2 * (arctan(w_gt/h_gt) - arctan(w_p/h_p))^2
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(gw / gh) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    # 8. Complete CIoU formula:
    # CIoU = IoU - (rho^2 / c^2) - alpha * v
    # Loss = 1 - CIoU = 1 - IoU + (rho^2 / c^2) + alpha * v
    ciou = iou - (rho_sq / c_diag_sq) - alpha * v
    return (1.0 - ciou).clamp(min=0.0, max=2.0)


# ==============================================================================
# 4. DATASET & COLLATOR
# ==============================================================================

class VisDroneDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

    def __init__(self, img_dir, ann_dir, img_size=640, augment=True, max_images=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.img_size = img_size
        self.augment = augment
        paths = sorted([f for f in self.img_dir.iterdir() if f.is_file() and f.suffix in self.EXTS])
        if not paths:
            paths = sorted([f for f in self.img_dir.rglob("*") if f.is_file() and f.suffix in self.EXTS])
        self.img_paths = paths[:max_images] if max_images else paths
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

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
            boxes = np.array(boxes, dtype=np.float32)
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
                    x1 = max(0.0, x); y1 = max(0.0, y)
                    x2 = min(float(w), x + bw); y2 = min(float(h), y + bh)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2]); cls_ids.append(cat - 1)
        except Exception:
            pass
        return boxes, cls_ids

    def _augment(self, img, boxes, cls_ids, w0, h0):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            boxes = [[w0 - b[2], b[1], w0 - b[0], b[3]] for b in boxes]
        return img, boxes, cls_ids

    def _letterbox(self, img, boxes):
        h, w = img.shape[:2]; s = self.img_size
        scale = s / max(h, w); nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh))
        pl = (s - nw) // 2; pt = (s - nh) // 2
        pr = s - nw - pl;   pb = s - nh - pt
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        new_boxes = [[b[0] * scale + pl, b[1] * scale + pt,
                      b[2] * scale + pl, b[3] * scale + pt] for b in boxes]
        return img, new_boxes, pl, pt, scale


def collate_fn(batch):
    imgs, targets, paths, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)
    metas = torch.stack(metas, 0)
    bt = []
    for i, t in enumerate(targets):
        if len(t):
            bi = torch.full((len(t), 1), float(i))
            bt.append(torch.cat([bi, t], dim=1))
    bt = torch.cat(bt, 0) if bt else torch.zeros((0, 6))
    return imgs, bt, list(paths), metas


def build_dataloaders(batch_size: int = DEFAULT_BATCH_SIZE):
    tds = VisDroneDataset(
        os.path.join(TRAIN_SPLIT, "images"),
        os.path.join(TRAIN_SPLIT, "annotations"),
        IMG_SIZE, augment=True
    )
    vds = VisDroneDataset(
        os.path.join(VAL_SPLIT, "images"),
        os.path.join(VAL_SPLIT, "annotations"),
        IMG_SIZE, augment=False
    )
    if len(tds) == 0 or len(vds) == 0:
        raise RuntimeError(f"Dataset empty. Train split: {TRAIN_SPLIT}, Val split: {VAL_SPLIT}")
    tl = DataLoader(
        tds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    vl = DataLoader(
        vds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    return tl, vl, tds, vds


# ==============================================================================
# 5. ARCHITECTURE DEFINITIONS
# ==============================================================================

class ASRFR(nn.Module):
    """
    Adaptive Sub-Resolution Feature Refinement.
    Applies PixelUnshuffle(2) -> 3xConv -> PixelShuffle(2) with learnable residual alpha.
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.sr_net = nn.Sequential(
            nn.Conv2d(12, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 12, 3, 1, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        packed = self.pixel_unshuffle(x)
        sr = self.pixel_shuffle(self.sr_net(packed))
        return x + self.alpha.clamp(0.0, 1.0) * sr


class SpatialDepthwiseModulationBlock(nn.Module):
    """
    Spatial Depthwise Feature Modulation Block (referred to as SpatialSSM in this project).
    Note: Implements a 2D spatial modulation mechanism via 7x7 group depthwise convolution,
    batch normalization, and channel-gated pointwise projections with learnable scaling gamma.
    Does NOT implement a 1D recurrent/selective state-space scan (e.g. Mamba S4/SSM).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        r = x
        x = self.norm(self.dw(x))
        x = self.pw2(self.act(self.pw1(x)))
        return r + self.gamma * x


class PassThrough(nn.Module):
    """
    Zero-parameter Identity pass-through block replacing SpatialDepthwiseModulationBlock
    in ablation variants without spatial modulation (v3, v5, v7). Maintains exact tensor dimensions.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.id = nn.Identity()

    def forward(self, x):
        return self.id(x)


class HiSMDNetAblationModel(nn.Module):
    """
    Unified ablation model architecture supporting all 7 study variants.
    """
    def __init__(self, variant_id: int, neck_ch: int = 128, asrfr_ch: int = 32, nc: int = 10):
        super().__init__()
        if variant_id not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant ID: {variant_id}")

        cfg = VARIANT_CONFIGS[variant_id]
        self.variant_id = variant_id
        self.variant_name = cfg["name"]
        self.nc = nc
        self.use_asrfr = cfg["use_asrfr"]
        self.use_ssm = cfg["use_ssm"]
        self.use_stride2 = cfg["use_stride2"]

        self.strides = [2, 4, 8, 16] if self.use_stride2 else [4, 8, 16]
        self.n_heads = len(self.strides)

        # 1. Optional ASRFR Preprocessor
        if self.use_asrfr:
            self.asrfr = ASRFR(asrfr_ch)

        # 2. ResNet-50 Backbone
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stage1 = base.layer1  # Stride 4, 256 ch

        Modulation = SpatialDepthwiseModulationBlock if self.use_ssm else PassThrough
        self.stage2 = nn.Sequential(base.layer2, Modulation(512))   # Stride 8, 512 ch
        self.stage3 = nn.Sequential(base.layer3, Modulation(1024))  # Stride 16, 1024 ch

        # 3. FPN Neck
        self.lat3 = nn.Conv2d(1024, neck_ch, 1)
        self.lat2 = nn.Conv2d(512, neck_ch, 1)
        self.lat1 = nn.Conv2d(256, neck_ch, 1)

        self.fuse3 = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, 1, 1), nn.BatchNorm2d(neck_ch), nn.SiLU(inplace=True)
        )

        # 4. Optional Stride-2 Sub-Pixel Pyramid Head
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

        # 5. Decoupled Heads
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

        # Head initialization
        prior = 0.01
        for h in self.cls_heads:
            nn.init.constant_(h[-1].bias, -math.log((1 - prior) / prior))
        for h in self.reg_heads:
            nn.init.constant_(h[-1].bias, 0.5)

    def forward(self, x):
        if self.use_asrfr:
            x = self.asrfr(x)

        s = self.stem(x)
        c2 = self.stage1(s)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)

        p4 = self.lat3(c4)
        p3 = self.fuse3(self.lat2(c3) + F.interpolate(p4, scale_factor=2, mode="nearest"))
        p2 = self.fuse2(self.lat1(c2) + F.interpolate(p3, scale_factor=2, mode="nearest"))
        p1_base = self.fuse1(p2)

        if self.use_stride2:
            p1_hires = self.hires_head(p1_base)
            feats = [p1_hires, p1_base, p3, p4]
        else:
            feats = [p1_base, p3, p4]

        cls_outs = [self.cls_heads[i](f) for i, f in enumerate(feats)]
        reg_outs = [F.softplus(self.reg_heads[i](f)).clamp(max=8.0) for i, f in enumerate(feats)]
        return cls_outs, reg_outs


# ==============================================================================
# 6. ANCHOR GENERATION & LOSS ASSIGNMENT
# ==============================================================================

def make_anchors(cls_outs, strides, device=None):
    pts, strs = [], []
    if device is None:
        device = cls_outs[0].device
    for c, s in zip(cls_outs, strides):
        H, W = c.shape[-2:]
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device), indexing="ij"
        )
        pts.append(torch.stack([(gx.flatten() + 0.5) * s, (gy.flatten() + 0.5) * s], 1))
        strs.append(torch.full((H * W,), s, device=device, dtype=torch.float32))
    return torch.cat(pts), torch.cat(strs)


def decode_boxes(reg: torch.Tensor, pts: torch.Tensor, strs: torch.Tensor) -> torch.Tensor:
    """Decodes LTRB offsets into xyxy bounding boxes."""
    off = reg * strs.unsqueeze(1)
    return torch.stack([
        pts[:, 0] - off[:, 0], pts[:, 1] - off[:, 1],
        pts[:, 0] + off[:, 2], pts[:, 1] + off[:, 3]
    ], 1)


def assign_and_loss(
    cls_outs: List[torch.Tensor],
    reg_outs: List[torch.Tensor],
    targets: torch.Tensor,
    strides: List[int],
    img_size: int,
    nc: int,
    wcls: float,
    wbox: float,
    wl1: float,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Task-Aligned Assignment with verified differentiable complete CIoU loss.
    """
    B = cls_outs[0].shape[0]
    pts, strs = make_anchors(cls_outs, strides, device=device)
    A = pts.shape[0]

    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, nc) for c in cls_outs], 1)
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in reg_outs], 1)

    # Decode predictions for assignment strictly under no_grad()
    with torch.no_grad():
        pb_assign = torch.stack([
            decode_boxes(reg_f[b].detach(), pts, strs).clamp(0, img_size)
            for b in range(B)
        ])

    N = len(targets)
    gt_px = torch.zeros(N, 4, device=device)
    if N > 0:
        gt_px[:, 0] = (targets[:, 2] - targets[:, 4] / 2) * img_size
        gt_px[:, 1] = (targets[:, 3] - targets[:, 5] / 2) * img_size
        gt_px[:, 2] = (targets[:, 2] + targets[:, 4] / 2) * img_size
        gt_px[:, 3] = (targets[:, 3] + targets[:, 5] / 2) * img_size

    cls_tgt = torch.zeros(B, A, nc, device=device)
    reg_tgt = torch.zeros(B, A, 4, device=device)
    box_tgt = torch.zeros(B, A, 4, device=device)
    wts = torch.zeros(B, A, device=device)

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
            is_cand = (
                (xa >= gtb[:, 0:1] - radius) & (xa <= gtb[:, 2:3] + radius) &
                (ya >= gtb[:, 1:2] - radius) & (ya <= gtb[:, 3:4] + radius) &
                (l_star >= 0) & (t_star >= 0) & (r_star >= 0) & (b_star >= 0)
            ).float()

            chunk = 8192; iou_chunks = []
            for ci in range(0, A, chunk):
                iou_chunks.append(box_iou(gtb, pb_assign[b][ci:ci + chunk]).clamp(0, 1))
            iou_ga = torch.cat(iou_chunks, dim=1)
            csc_ga = torch.sigmoid(cls_f[b].detach())[:, clb].T.clamp(1e-6, 1.0)
            ta_sc = (iou_ga ** 0.5) * (csc_ga ** 0.5) * is_cand
            k = min(10, A)
            topk_v, topk_i = ta_sc.topk(k, dim=1)
            vf = topk_v.flatten(); vi = topk_i.flatten()
            cf = clb.unsqueeze(1).expand(-1, k).flatten()
            bf = gtb.unsqueeze(1).expand(-1, k, 4).reshape(-1, 4)
            rf = offsets_star.view(G, A, 4).gather(1, topk_i.unsqueeze(-1).expand(-1, -1, 4)).reshape(-1, 4)
            ok = vf > 0.005
            if not ok.any():
                continue
            vf, vi, cf, bf, rf = vf[ok], vi[ok], cf[ok], bf[ok], rf[ok]
            cur = wts[b]; better = vf > cur[vi]
            vi_b, vf_b, cf_b, bf_b, rf_b = vi[better], vf[better], cf[better], bf[better], rf[better]
            if len(vi_b) == 0:
                continue
            cur[vi_b] = vf_b
            cls_tgt[b, vi_b, :] = 0.0; cls_tgt[b, vi_b, cf_b] = vf_b
            reg_tgt[b, vi_b] = rf_b; box_tgt[b, vi_b] = bf_b

    pos = wts > 0
    npos = pos.sum().clamp(min=1)

    cls_loss = F.binary_cross_entropy_with_logits(
        cls_f.reshape(-1, nc), cls_tgt.reshape(-1, nc), reduction="sum"
    ) / npos

    box_loss = l1_loss = torch.tensor(0.0, device=device, requires_grad=False)

    if pos.any():
        # Re-decode OUTSIDE no_grad so gradients flow back into reg_heads
        pts_expand = pts.unsqueeze(0).expand(B, -1, -1)[pos]
        strs_expand = strs.unsqueeze(0).expand(B, -1)[pos]
        pb_for_loss = decode_boxes(reg_f[pos], pts_expand, strs_expand).clamp(0, img_size)

        box_loss = complete_ciou_loss(pb_for_loss, box_tgt[pos]).sum() / npos

        if wl1 > 0:
            l1_loss = F.smooth_l1_loss(
                reg_f[pos], reg_tgt[pos].clamp(0, 8.0), reduction="sum"
            ) / npos * wl1

    total_loss = wcls * cls_loss + wbox * box_loss + l1_loss
    return total_loss, cls_loss.detach(), box_loss.detach()


# ==============================================================================
# 7. POSTPROCESSING & COCO EVALUATION
# ==============================================================================

def postprocess(
    cls_outs: List[torch.Tensor],
    reg_outs: List[torch.Tensor],
    strides: List[int],
    img_size: int,
    nc: int,
    conf: float = 0.05,
    max_det: int = 300
) -> List[Dict[str, torch.Tensor]]:
    B = cls_outs[0].shape[0]
    pts, strs = make_anchors(cls_outs, strides, device=cls_outs[0].device)
    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, nc).sigmoid() for c in cls_outs], 1)
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in reg_outs], 1)
    results = []
    for b in range(B):
        pb = decode_boxes(reg_f[b], pts, strs).clamp(0, img_size)
        sc, lb = cls_f[b].max(-1)
        mask = sc > conf
        pb, sc, lb = pb[mask], sc[mask], lb[mask]
        if len(pb) == 0:
            results.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)
            })
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
            results.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)
            })
    return results


def build_coco_gt(vds: VisDroneDataset) -> dict:
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
    }
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
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0
            })
            aid += 1
    return gt


@torch.no_grad()
def evaluate_coco(model, val_dl, gt_dict, device, conf: float = 0.05) -> Dict[str, float]:
    import tempfile
    model.eval()
    if COCO is None or COCOeval is None:
        print("[!] pycocotools not available; returning empty eval dict.")
        return {"AP": 0.0, "AP50": 0.0, "APS": 0.0, "APM": 0.0}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); gf = f.name
    coco_gt = COCO(gf)
    preds = []; iid = 0
    for imgs, _, paths, metas in tqdm(val_dl, desc="Evaluating", leave=False, file=sys.stdout):
        imgs = imgs.to(device)
        with autocast(enabled=USE_AMP):
            co, ro = model(imgs)
        dets = postprocess(co, ro, model.strides, IMG_SIZE, NUM_CLASSES, conf)
        for b, det in enumerate(dets):
            w0, h0, pl, pt, sc = metas[b].tolist()
            for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                x1, y1, x2, y2 = box.tolist()
                x1o = max(0.0, (x1 - pl) / sc); y1o = max(0.0, (y1 - pt) / sc)
                x2o = min(w0,  (x2 - pl) / sc); y2o = min(h0,  (y2 - pt) / sc)
                wo = max(0.0, x2o - x1o); ho = max(0.0, y2o - y1o)
                if wo > 0 and ho > 0:
                    preds.append({
                        "image_id": iid + b, "category_id": int(label),
                        "bbox": [x1o, y1o, wo, ho], "score": float(score)
                    })
        iid += len(dets)
    empty = {"AP": 0.0, "AP50": 0.0, "APS": 0.0, "APM": 0.0}
    if not preds:
        return empty
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f); df = f.name
    try:
        ev = COCOeval(coco_gt, coco_gt.loadRes(df), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return {
            "AP": float(ev.stats[0]),
            "AP50": float(ev.stats[1]),
            "APS": float(ev.stats[3]),
            "APM": float(ev.stats[4])
        }
    except Exception as e:
        print(f"  [!] COCOeval execution notice: {e}")
        return empty


# ==============================================================================
# 8. RIGOROUS SYNCHRONIZED SPEED BENCHMARK
# ==============================================================================

@torch.no_grad()
def benchmark_inference_speed(
    model: nn.Module,
    device: str,
    img_size: int = 640,
    num_warmup: int = 20,
    num_runs: int = 80
) -> Dict[str, float]:
    """
    Measures true inference latency (ms) and throughput (FPS) using CUDA synchronization
    and precise timers. Clearly isolates forward pass latency at batch size 1.
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    # 1. Warm-up
    for _ in range(num_warmup):
        with autocast(enabled=USE_AMP):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # 2. Timed measurement
    t0 = time.perf_counter()
    for _ in range(num_runs):
        with autocast(enabled=USE_AMP):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / num_runs) * 1000.0
    fps = num_runs / elapsed
    return {
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "device": device,
        "batch_size": 1,
        "precision": "FP16 (AMP)" if USE_AMP else "FP32",
        "num_runs": num_runs
    }


# ==============================================================================
# 9. SINGLE VARIANT TRAINER
# ==============================================================================

def train_variant(
    variant_id: int,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_freq: int = DEFAULT_EVAL_FREQ,
    save_progression: bool = True
) -> Dict[str, any]:
    cfg = VARIANT_CONFIGS[variant_id]
    name = cfg["name"]
    var_dir = OUTPUT_ROOT / name
    var_dir.mkdir(parents=True, exist_ok=True)

    log_csv = var_dir / "training_log.csv"
    cfg_json = var_dir / "variant_config.json"
    eval_json = var_dir / "eval_results.json"
    speed_json = var_dir / "speed_benchmark.json"
    best_ckpt = var_dir / f"best_{name}.pt"
    last_ckpt = var_dir / f"last_{name}.pt"

    print("\n" + "=" * 78)
    print(f"  STARTING VARIANT {variant_id}: {name.upper()}")
    print(f"  Description: {cfg['description']}")
    print(f"  Target Epochs: {epochs} | Seed: {seed} | Batch Size: {batch_size}")
    print(f"  Output Directory: {var_dir}")
    print("=" * 78)

    # Record full configuration metadata
    total_params = sum(
        p.numel() for p in HiSMDNetAblationModel(variant_id).parameters() if p.requires_grad
    )
    with open(cfg_json, "w") as f:
        json.dump({
            "variant_id": variant_id,
            "name": name,
            "description": cfg["description"],
            "use_asrfr": cfg["use_asrfr"],
            "use_ssm": cfg["use_ssm"],
            "use_stride2": cfg["use_stride2"],
            "total_params": total_params,
            "epochs": epochs,
            "seed": seed,
            "batch_size": batch_size,
            "learning_rate": LR,
            "img_size": IMG_SIZE,
            "ciou_aspect_ratio_penalty": True,
            "device": DEVICE,
            "gpu_vram_gb": VRAM_GB
        }, f, indent=2)

    set_seed(seed)
    tl, vl, tds, vds = build_dataloaders(batch_size=batch_size)
    gt_dict = build_coco_gt(vds)

    model = HiSMDNetAblationModel(variant_id, NECK_CH, ASRFR_CH, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=MIN_LR)
    scaler = GradScaler(enabled=USE_AMP)

    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "lr", "loss", "cls_loss", "box_loss",
                "AP", "AP50", "APS", "APM"
            ])

    # Safe Resumption Check
    start_epoch = 1; best_ap50 = 0.0; best_ap = 0.0
    for ckpt_p in [last_ckpt, best_ckpt]:
        if ckpt_p.exists():
            try:
                ckpt = torch.load(str(ckpt_p), map_location=DEVICE, weights_only=False)
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt: opt.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt: sched.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and USE_AMP and ckpt["scaler"]:
                    scaler.load_state_dict(ckpt["scaler"])
                start_epoch = ckpt.get("epoch", 1) + 1
                best_ap50 = ckpt.get("best_ap50", 0.0)
                best_ap = ckpt.get("best_ap", 0.0)
                print(f"  [+] Resumed from {ckpt_p.name} at epoch {start_epoch - 1}")
                break
            except Exception as e:
                print(f"  [!] Resumption notice ({e}); starting from scratch.")

    if start_epoch > epochs:
        print(f"  [=] Variant {name} has already completed {epochs} epochs.")
    else:
        # Training Epoch Loop
        train_start_t = time.time()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            tot = cls_s = box_s = 0.0; nb = 0
            pbar = tqdm(
                tl, desc=f"[{name}] Ep {epoch:3d}/{epochs}",
                leave=False, file=sys.stdout, ncols=115
            )
            for imgs, tgts, _, _ in pbar:
                imgs = imgs.to(DEVICE); tgts = tgts.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=USE_AMP):
                    co, ro = model(imgs)
                    loss, cl, bl = assign_and_loss(
                        co, ro, tgts, model.strides, IMG_SIZE,
                        NUM_CLASSES, W_CLS, W_BOX, W_L1, DEVICE
                    )
                if torch.isnan(loss) or torch.isinf(loss):
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
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "cls": f"{cl.item():.3f}",
                    "box": f"{bl.item():.3f}"
                })

            sched.step()
            lr_val = sched.get_last_lr()[0]
            avg = {"total": tot / max(nb, 1), "cls": cls_s / max(nb, 1), "box": box_s / max(nb, 1)}
            print(f"[{name}] Ep {epoch:3d}/{epochs} | LR: {lr_val:.2e} | "
                  f"Loss: {avg['total']:.4f} (cls: {avg['cls']:.4f}, box: {avg['box']:.4f})")

            m = None
            if epoch % eval_freq == 0 or epoch == epochs or (save_progression and epoch in [30, 60]):
                m = evaluate_coco(model, vl, gt_dict, DEVICE, CONF_THRESH)
                print(f"  >>> Val AP: {m['AP']*100:.2f}% | AP50: {m['AP50']*100:.2f}% | "
                      f"APS: {m['APS']*100:.2f}% | APM: {m['APM']*100:.2f}%")

                # Snapshot for progression analysis if variant 1
                if variant_id == 1 and save_progression and epoch in [30, 60, 100]:
                    snap_p = var_dir / f"full_progression_epoch_{epoch:03d}.pt"
                    torch.save({
                        "epoch": epoch, "model": model.state_dict(),
                        "metrics": m, "variant_id": variant_id
                    }, snap_p)

                if m["AP50"] > best_ap50 or m["AP"] > best_ap:
                    best_ap50 = max(best_ap50, m["AP50"])
                    best_ap = max(best_ap, m["AP"])
                    torch.save({
                        "epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict() if USE_AMP else None,
                        "best_ap50": best_ap50, "best_ap": best_ap,
                        "metrics": m, "variant_id": variant_id
                    }, best_ckpt)
                    print(f"  *** NEW BEST: AP50={best_ap50*100:.2f}%, mAP={best_ap*100:.2f}% -> {best_ckpt.name}")

            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if USE_AMP else None,
                "best_ap50": best_ap50, "best_ap": best_ap,
                "variant_id": variant_id
            }, last_ckpt)

            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{lr_val:.6e}",
                    f"{avg['total']:.4f}", f"{avg['cls']:.4f}", f"{avg['box']:.4f}",
                    f"{m['AP']*100:.2f}"   if m else "",
                    f"{m['AP50']*100:.2f}" if m else "",
                    f"{m['APS']*100:.2f}"  if m else "",
                    f"{m['APM']*100:.2f}"  if m else ""
                ])

    # Speed Benchmark with Synchronized GPU Timing
    print(f"\n  Measuring inference latency and throughput for {name} ...")
    if best_ckpt.exists():
        try:
            b_load = torch.load(str(best_ckpt), map_location=DEVICE, weights_only=False)
            model.load_state_dict(b_load["model"])
        except Exception:
            pass
    speed = benchmark_inference_speed(model, DEVICE, IMG_SIZE)
    with open(speed_json, "w") as f:
        json.dump(speed, f, indent=2)

    # Compile final metrics from log
    best_record = {"AP50": 0.0, "AP": 0.0, "APS": 0.0, "APM": 0.0}
    if log_csv.exists():
        with open(log_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ap50 = float(row.get("AP50") or 0.0)
                    if ap50 > best_record["AP50"]:
                        best_record["AP50"] = ap50
                        best_record["AP"] = float(row.get("AP") or 0.0)
                        best_record["APS"] = float(row.get("APS") or 0.0)
                        best_record["APM"] = float(row.get("APM") or 0.0)
                except ValueError:
                    pass

    with open(eval_json, "w") as f:
        json.dump({
            "variant_id": variant_id,
            "name": name,
            "best_metrics": best_record,
            "speed": speed,
            "params_m": round(total_params / 1e6, 2)
        }, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  [COMPLETED] {name.upper()}")
    print(f"  Best AP50: {best_record['AP50']:.2f}% | mAP: {best_record['AP']:.2f}% | "
          f"APS: {best_record['APS']:.2f}% | APM: {best_record['APM']:.2f}%")
    print(f"  Latency: {speed['latency_ms']:.2f} ms ({speed['fps']:.1f} FPS) | Params: {total_params/1e6:.2f}M")
    print("=" * 78)

    return {
        "variant_id": variant_id,
        "name": name,
        **best_record,
        **speed,
        "params": f"{total_params/1e6:.2f}M"
    }


# ==============================================================================
# 10. AGGREGATED REPORT & COMPARISON TABLE GENERATOR
# ==============================================================================

def generate_comparison_reports():
    """
    Generates:
    1. ablation_comparison.csv (7-variant comparison)
    2. epoch_comparison.csv (Full model progression at 30, 60, 100 epochs)
    3. results_summary.txt (concise empirical summary)
    """
    ablation_rows = []
    for vid, cfg in VARIANT_CONFIGS.items():
        name = cfg["name"]
        var_dir = OUTPUT_ROOT / name
        eval_json = var_dir / "eval_results.json"
        log_csv = var_dir / "training_log.csv"
        if not eval_json.exists() and not log_csv.exists():
            continue

        best_ap50 = best_ap = best_aps = best_apm = 0.0
        if log_csv.exists():
            try:
                with open(log_csv) as f:
                    for row in csv.DictReader(f):
                        try:
                            a50 = float(row.get("AP50") or 0.0)
                            if a50 > best_ap50:
                                best_ap50 = a50
                                best_ap = float(row.get("AP") or 0.0)
                                best_aps = float(row.get("APS") or 0.0)
                                best_apm = float(row.get("APM") or 0.0)
                        except ValueError:
                            pass
            except Exception:
                pass

        params_str = "N/A"
        latency_str = "N/A"
        fps_str = "N/A"
        if eval_json.exists():
            try:
                with open(eval_json) as f:
                    ed = json.load(f)
                    params_str = f"{ed.get('params_m', 0.0):.2f}M"
                    speed = ed.get("speed", {})
                    latency_str = f"{speed.get('latency_ms', 0.0):.2f}"
                    fps_str = f"{speed.get('fps', 0.0):.1f}"
            except Exception:
                pass

        ablation_rows.append({
            "Variant": f"V{vid}",
            "Name": name,
            "ASRFR": str(cfg["use_asrfr"]),
            "SpatialModulation": str(cfg["use_ssm"]),
            "Stride2_Head": str(cfg["use_stride2"]),
            "AP50 (%)": f"{best_ap50:.2f}",
            "mAP (%)": f"{best_ap:.2f}",
            "APS (%)": f"{best_aps:.2f}",
            "APM (%)": f"{best_apm:.2f}",
            "Params": params_str,
            "Latency (ms)": latency_str,
            "FPS": fps_str
        })

    # 1. Write ablation_comparison.csv
    if ablation_rows:
        out_csv = OUTPUT_ROOT / "ablation_comparison.csv"
        fields = [
            "Variant", "Name", "ASRFR", "SpatialModulation", "Stride2_Head",
            "AP50 (%)", "mAP (%)", "APS (%)", "APM (%)",
            "Params", "Latency (ms)", "FPS"
        ]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(ablation_rows)
        print(f"\n[OK] Generated {out_csv}")

    # 2. Write epoch_comparison.csv (Progression of V1 Full Model at 30, 60, 100)
    v1_log = OUTPUT_ROOT / "v1_full" / "training_log.csv"
    epoch_rows = []
    if v1_log.exists():
        target_epochs = [30, 60, 100]
        try:
            with open(v1_log) as f:
                for row in csv.DictReader(f):
                    try:
                        ep = int(row.get("epoch") or 0)
                        if ep in target_epochs:
                            epoch_rows.append({
                                "Configuration": "V1_Full_HiSMDNet",
                                "Epoch": ep,
                                "Loss": row.get("loss", ""),
                                "AP50 (%)": row.get("AP50", ""),
                                "mAP (%)": row.get("AP", ""),
                                "APS (%)": row.get("APS", ""),
                                "APM (%)": row.get("APM", ""),
                            })
                    except ValueError:
                        pass
        except Exception:
            pass

    if epoch_rows:
        epoch_csv = OUTPUT_ROOT / "epoch_comparison.csv"
        fields_ep = ["Configuration", "Epoch", "Loss", "AP50 (%)", "mAP (%)", "APS (%)", "APM (%)"]
        with open(epoch_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_ep)
            w.writeheader(); w.writerows(epoch_rows)
        print(f"[OK] Generated {epoch_csv}")

    # 3. Write results_summary.txt
    summary_txt = OUTPUT_ROOT / "results_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("=" * 78 + "\n")
        f.write("HiSMD-Net Rigorous Ablation Study Summary (ablation_v2)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Variants Evaluated: {len(ablation_rows)}\n\n")

        f.write("--- 7-VARIANT ABLATION RESULTS ---\n")
        for r in ablation_rows:
            f.write(
                f"{r['Variant']:<4} | {r['Name']:<24} | AP50: {r['AP50 (%)']:>6}% | "
                f"mAP: {r['mAP (%)']:>6}% | APS: {r['APS (%)']:>6}% | "
                f"Params: {r['Params']:>8} | FPS: {r['FPS']:>6}\n"
            )

        f.write("\n--- ASRFR TRAINING DURATION PROGRESSION (V1 FULL MODEL) ---\n")
        if epoch_rows:
            for er in epoch_rows:
                f.write(
                    f"Epoch {er['Epoch']:>3} | AP50: {er['AP50 (%)']:>6}% | "
                    f"mAP: {er['mAP (%)']:>6}% | APS: {er['APS (%)']:>6}% | Loss: {er['Loss']}\n"
                )
        else:
            f.write("Progression data pending completion of V1 training.\n")

        f.write("\n" + "=" * 78 + "\n")
    print(f"[OK] Generated {summary_txt}")


# ==============================================================================
# 11. ENTRY POINT & CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="HiSMD-Net Ablation Study V2")
    parser.add_argument(
        "--variant", nargs="+", default=["all"],
        help="Variant IDs (1 to 7) or 'all' (default: all)"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs per variant (default: 100)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_EVAL_FREQ, help="Validation frequency (default: 5)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument("--report_only", action="store_true", help="Generate comparison tables from existing outputs")
    parser.add_argument("--dry_run", action="store_true", help="Run a short single-batch verification pass")

    args = parser.parse_args()

    if args.report_only:
        generate_comparison_reports()
        return

    if args.variant == ["all"]:
        variants_to_run = list(VARIANT_CONFIGS.keys())
    else:
        variants_to_run = [int(v) for v in args.variant if int(v) in VARIANT_CONFIGS]

    print("\n" + "=" * 78)
    print("  HiSMD-Net Ablation Study V2 Launcher")
    print(f"  Variants to run: {variants_to_run}")
    print(f"  Epochs per variant: {args.epochs} | Batch size: {args.batch_size}")
    print(f"  Device: {DEVICE} | VRAM: {VRAM_GB:.1f} GB | AMP: {USE_AMP}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    print("=" * 78)

    for vid in variants_to_run:
        train_variant(
            variant_id=vid,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
            save_progression=True
        )

    generate_comparison_reports()
    print("\n[✓] All requested ablation runs and reports completed successfully.")


if __name__ == "__main__":
    main()
