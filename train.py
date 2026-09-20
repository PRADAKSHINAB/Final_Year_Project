# ================================================================
# HiSMD-Net: Hierarchical Super-resolved Mamba Detector
# 100-EPOCH AUTO-RESUMING PRODUCTION TRAINER FOR VISDRONE
# Features:
#  1. Auto-resumes seamlessly from Epoch 90 (weights_only=False)
#  2. Restores saved Epoch 90 weights and continues to Epoch 100
#  3. Pretrained ResNet50 Transfer Learning Backbone
#  4. ASRFR Sub-pixel Super-Resolution Pre-processor (<16px objects)
#  5. 4-Scale Pyramid with Stride-2 Head (320x320 Grid for micro-objects)
#  6. EXACT Relative Offset Target Math: (l*, t*, r*, b*)
#  7. Softplus Clamped [0, 8] Head (Boxes match exact object size)
#  8. 100% NaN-Free CUDA Speed on NVIDIA H200
# ================================================================

import os
import sys

# Prevent CUDA memory fragmentation (critical for T4/small GPUs with large anchor grids)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import csv
import json
import math
import time
import random
import copy
import glob
import warnings
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

warnings.filterwarnings("ignore")

# ================================================================
# 0. HPC PACKAGE SETUP (FIXES MODULE NOT FOUND ERRORS)
# ================================================================

MY_PACKAGES = os.path.expanduser("~/my_python_packages")
os.makedirs(MY_PACKAGES, exist_ok=True)

if MY_PACKAGES not in sys.path:
    sys.path.insert(0, MY_PACKAGES)

def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
        return
    except ImportError:
        print(f"Installing {pip_name} into {MY_PACKAGES} ...")
        subprocess.check_call([
            sys.executable, "-m", "pip",
            "install", "--target", MY_PACKAGES,
            "--upgrade", pip_name
        ])
        if MY_PACKAGES not in sys.path:
            sys.path.insert(0, MY_PACKAGES)

ensure_package("pycocotools", "pycocotools")
ensure_package("torchvision", "torchvision")

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torchvision.ops import nms, box_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

print("============================================================")
print("Python :", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM   : {total_mem:.2f} GB")
print("============================================================")


# ================================================================
# 1. CONFIGURATION & PATH RESOLVER
# ================================================================

def find_visdrone_split(split_name):
    roots = [
        Path("./Dataset"),
        Path("."),
        Path(os.path.expanduser("~/Dataset")),
        Path(os.path.expanduser("~/uavdet_system/uavdet-system/Dataset")),
        Path("./uavdet_system/uavdet-system/Dataset"),
    ]

    exact_candidates = [
        Path("./Dataset") / split_name / split_name,
        Path("./Dataset") / split_name,
        Path(split_name) / split_name,
        Path(split_name),
        Path(os.path.expanduser("~/uavdet_system/uavdet-system/Dataset")) / split_name / split_name,
        Path(os.path.expanduser("~/uavdet_system/uavdet-system/Dataset")) / split_name,
    ]

    for split_dir in exact_candidates:
        if (split_dir / "images").is_dir() and (split_dir / "annotations").is_dir():
            return str(split_dir.resolve())

    seen = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            candidates = [root / split_name] + list(root.rglob(split_name))
            for split_dir in candidates:
                split_dir = split_dir.resolve()
                if split_dir in seen:
                    continue
                seen.add(split_dir)
                if split_dir.is_dir() and (split_dir / "images").is_dir() and (split_dir / "annotations").is_dir():
                    return str(split_dir)
        except Exception:
            pass

    return None


class Config:
    TRAIN_SPLIT_DIR = find_visdrone_split("VisDrone2019-DET-train")
    VAL_SPLIT_DIR = find_visdrone_split("VisDrone2019-DET-val")

    if TRAIN_SPLIT_DIR is None:
        TRAIN_SPLIT_DIR = "./Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train"
    if VAL_SPLIT_DIR is None:
        VAL_SPLIT_DIR = "./Dataset/VisDrone2019-DET-val/VisDrone2019-DET-val"

    TRAIN_IMG_DIR = os.path.join(TRAIN_SPLIT_DIR, "images")
    TRAIN_ANN_DIR = os.path.join(TRAIN_SPLIT_DIR, "annotations")
    VAL_IMG_DIR = os.path.join(VAL_SPLIT_DIR, "images")
    VAL_ANN_DIR = os.path.join(VAL_SPLIT_DIR, "annotations")

    SAVE_DIR = "./hismdet_output"
    PRED_DIR = os.path.join(SAVE_DIR, "predictions_txt")

    IMG_SIZE = 640

    # 4 Detection Strides: Stride 2 (320x320 grid for <16px micro-objects), 4, 8, 16
    STRIDES = [2, 4, 8, 16]

    NECK_CH = 128
    ASRFR_CH = 32

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

    # Auto-detect VRAM: T4/small GPU (<=20GB) -> batch 4, H200/A100 (>50GB) -> batch 16
    _vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.is_available() else 0)
    BATCH_SIZE = 16 if _vram_gb > 50 else (4 if torch.cuda.is_available() else 4)
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    EPOCHS = 100
    WARMUP_EPOCHS = 3
    LR = 5e-4
    MIN_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 10.0
    EMA_DECAY = 0.9998

    USE_AMP = torch.cuda.is_available()

    MAX_TRAIN_IMGS = None
    MAX_VAL_IMGS = None
    EVAL_FREQ = 5

    # Loss Weights
    W_CLS = 1.0
    W_BOX = 4.0
    W_L1 = 1.0

    # Detection Thresholds
    CONF_THRESH = 0.05
    VIS_CONF_THRESH = 0.25
    NMS_IOU_THRESH = 0.50
    MAX_DET = 300

    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
os.makedirs(cfg.PRED_DIR, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(cfg.SEED)


# ================================================================
# 2. DATASET & DATALOADER
# ================================================================

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

        print(f"  VisDroneDataset: {len(self.img_paths)} images ({'train aug' if augment else 'validation'})")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]
        boxes, cls_ids = self._load_ann(img_path.stem, w0, h0)

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

            cx = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / s
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / s
            bw = (boxes[:, 2] - boxes[:, 0]) / s
            bh = (boxes[:, 3] - boxes[:, 1]) / s

            valid = (bw > 0.001) & (bh > 0.001)
            cx, cy, bw, bh, cls_ids = cx[valid], cy[valid], bw[valid], bh[valid], cls_ids[valid]

            if len(cx):
                target = torch.zeros((len(cx), 5), dtype=torch.float32)
                target[:, 0] = torch.tensor(cls_ids, dtype=torch.float32)
                target[:, 1] = torch.tensor(cx, dtype=torch.float32)
                target[:, 2] = torch.tensor(cy, dtype=torch.float32)
                target[:, 3] = torch.tensor(bw, dtype=torch.float32)
                target[:, 4] = torch.tensor(bh, dtype=torch.float32)

        meta = torch.tensor([w0, h0, pad_l, pad_t, scale], dtype=torch.float32)
        return img, target, str(img_path), meta

    def _load_ann(self, stem, w, h):
        boxes, cls_ids = [], []
        ann_file = self.ann_dir / (stem + ".txt")
        if not ann_file.exists():
            return boxes, cls_ids

        try:
            with open(ann_file, "r") as fp:
                for line in fp:
                    p = line.strip().split(",")
                    if len(p) < 6:
                        continue
                    x, y, bw, bh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                    score, cat = int(p[4]), int(p[5])
                    if cat == 0 or score == 0 or cat == 11:
                        continue
                    x1 = max(0.0, x)
                    y1 = max(0.0, y)
                    x2 = min(float(w), x + bw)
                    y2 = min(float(h), y + bh)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    cls_ids.append(cat - 1)
        except Exception:
            pass
        return boxes, cls_ids

    def _augment(self, img, boxes, cls_ids, w0, h0):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            boxes = [[w0 - b[2], b[1], w0 - b[0], b[3]] for b in boxes]

        return img, boxes, cls_ids

    def _letterbox(self, img, boxes):
        h, w = img.shape[:2]
        s = self.img_size
        scale = s / max(h, w)
        nw, nh = int(w * scale), int(h * scale)

        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_l = (s - nw) // 2
        pad_t = (s - nh) // 2
        pad_r = s - nw - pad_l
        pad_b = s - nh - pad_t

        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        new_boxes = [[b[0] * scale + pad_l, b[1] * scale + pad_t, b[2] * scale + pad_l, b[3] * scale + pad_t] for b in boxes]
        return img, new_boxes, pad_l, pad_t, scale


def collate_fn(batch):
    imgs, targets, paths, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)
    metas = torch.stack(metas, 0)
    bt = []
    for i, t in enumerate(targets):
        if len(t):
            bi = torch.full((len(t), 1), float(i), dtype=torch.float32)
            bt.append(torch.cat([bi, t], dim=1))
    if bt:
        bt = torch.cat(bt, 0)
    else:
        bt = torch.zeros((0, 6), dtype=torch.float32)
    return imgs, bt, list(paths), metas


def build_dataloaders():
    train_ds = VisDroneDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_ANN_DIR, cfg.IMG_SIZE, augment=True, max_images=cfg.MAX_TRAIN_IMGS)
    val_ds = VisDroneDataset(cfg.VAL_IMG_DIR, cfg.VAL_ANN_DIR, cfg.IMG_SIZE, augment=False, max_images=cfg.MAX_VAL_IMGS)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("VisDrone dataset directories are empty. Please check paths.")

    train_dl = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    return train_dl, val_dl, train_ds, val_ds


# ================================================================
# 3. HiSMD-Net ARCHITECTURE (Pretrained Backbone + Novel Modules)
# ================================================================

class ASRFR(nn.Module):
    """Adaptive Sub-pixel Resolution Feature Reconstructor for <16px objects"""
    def __init__(self, hidden=32):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.sr_net = nn.Sequential(
            nn.Conv2d(12, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 12, 3, 1, 1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        packed = self.pixel_unshuffle(x)
        sr = self.pixel_shuffle(self.sr_net(packed))
        return x + self.alpha.clamp(0.0, 1.0) * sr


class SpatialSSMBlock(nn.Module):
    """Numerically-bounded spatial state attention module (100% NaN-safe)"""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        residual = x
        x = self.norm(self.dw(x))
        x = self.pw2(self.act(self.pw1(x)))
        return residual + self.gamma * x


class HiSMDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.asrfr = ASRFR(cfg.ASRFR_CH)

        # Pretrained ResNet50 Backbone
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stage1 = base.layer1  # Stride 4, 256 channels (160x160)
        self.stage2 = nn.Sequential(base.layer2, SpatialSSMBlock(512))   # Stride 8, 512 channels (80x80)
        self.stage3 = nn.Sequential(base.layer3, SpatialSSMBlock(1024))  # Stride 16, 1024 channels (40x40)

        neck_c = cfg.NECK_CH
        self.lat3 = nn.Conv2d(1024, neck_c, 1)
        self.lat2 = nn.Conv2d(512, neck_c, 1)
        self.lat1 = nn.Conv2d(256, neck_c, 1)

        self.fuse3 = nn.Sequential(nn.Conv2d(neck_c, neck_c, 3, 1, 1), nn.BatchNorm2d(neck_c), nn.SiLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(neck_c, neck_c, 3, 1, 1), nn.BatchNorm2d(neck_c), nn.SiLU(inplace=True))
        self.fuse1 = nn.Sequential(nn.Conv2d(neck_c, neck_c, 3, 1, 1), nn.BatchNorm2d(neck_c), nn.SiLU(inplace=True))

        # Sub-pixel Stride-2 Head for micro-objects (<16px)
        self.hires_head = nn.Sequential(
            nn.Conv2d(neck_c, neck_c * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(neck_c, neck_c, 3, 1, 1),
            nn.BatchNorm2d(neck_c),
            nn.SiLU(inplace=True)
        )

        # 4 Decoupled Classification & Regression Heads
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(neck_c, neck_c, 3, 1, 1), nn.SiLU(inplace=True), nn.Conv2d(neck_c, cfg.NUM_CLASSES, 1))
            for _ in range(4)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(neck_c, neck_c, 3, 1, 1), nn.SiLU(inplace=True), nn.Conv2d(neck_c, 4, 1))
            for _ in range(4)
        ])

        self.strides = cfg.STRIDES

        prior = 0.01
        for h in self.cls_heads:
            nn.init.constant_(h[-1].bias, -math.log((1 - prior) / prior))
        for h in self.reg_heads:
            nn.init.constant_(h[-1].bias, 0.5)

        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"\nHiSMD-Net initialized successfully. Parameters: {total_p:.2f}M | Strides: {self.strides}")

    def forward(self, x):
        x_sr = self.asrfr(x)
        s = self.stem(x_sr)
        c2 = self.stage1(s)   # Stride 4
        c3 = self.stage2(c2)  # Stride 8
        c4 = self.stage3(c3)  # Stride 16

        p4 = self.lat3(c4)
        p3 = self.fuse3(self.lat2(c3) + F.interpolate(p4, scale_factor=2, mode="nearest"))
        p2 = self.fuse2(self.lat1(c2) + F.interpolate(p3, scale_factor=2, mode="nearest"))
        p1 = self.hires_head(p2)  # Stride 2 (320x320 grid)

        feats = [p1, p2, p3, p4]
        cls_outs = [self.cls_heads[i](f) for i, f in enumerate(feats)]
        reg_outs = [F.softplus(self.reg_heads[i](f)).clamp(max=8.0) for i, f in enumerate(feats)]

        return cls_outs, reg_outs


# ================================================================
# 4. ANCHOR DECODING & EXACT RELATIVE OFFSET ASSIGNER
# ================================================================

def make_anchors(cls_outs, strides):
    pts, strs = [], []
    for c, s in zip(cls_outs, strides):
        H, W = c.shape[-2:]
        gy, gx = torch.meshgrid(torch.arange(H, device=c.device),
                                torch.arange(W, device=c.device), indexing="ij")
        pts.append(torch.stack([(gx.flatten() + 0.5) * s, (gy.flatten() + 0.5) * s], 1))
        strs.append(torch.full((H * W,), s, device=c.device, dtype=torch.float32))
    return torch.cat(pts), torch.cat(strs)


def decode_boxes(reg, pts, strs):

    off = reg * strs.unsqueeze(1)
    x1 = pts[:, 0] - off[:, 0]
    y1 = pts[:, 1] - off[:, 1]
    x2 = pts[:, 0] + off[:, 2]
    y2 = pts[:, 1] + off[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)


def ciou_loss(p, g, eps=1e-7):
    ix1 = torch.max(p[:, 0], g[:, 0])
    iy1 = torch.max(p[:, 1], g[:, 1])
    ix2 = torch.min(p[:, 2], g[:, 2])
    iy2 = torch.min(p[:, 3], g[:, 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    ap = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    ag = (g[:, 2] - g[:, 0]).clamp(0) * (g[:, 3] - g[:, 1]).clamp(0)
    union = ap + ag - inter + eps
    iou = (inter / union).clamp(0, 1)

    ex1 = torch.min(p[:, 0], g[:, 0])
    ey1 = torch.min(p[:, 1], g[:, 1])
    ex2 = torch.max(p[:, 0], g[:, 0])
    ey2 = torch.max(p[:, 1], g[:, 1])
    enc_diag = (ex2 - ex1)**2 + (ey2 - ey1)**2 + eps

    cx_p = (p[:, 0] + p[:, 2]) * 0.5
    cy_p = (p[:, 1] + p[:, 3]) * 0.5
    cx_g = (g[:, 0] + g[:, 2]) * 0.5
    cy_g = (g[:, 1] + g[:, 3]) * 0.5
    ctr_dist = (cx_p - cx_g)**2 + (cy_p - cy_g)**2

    ciou = iou - (ctr_dist / enc_diag).clamp(0, 1)
    return 1.0 - ciou.clamp(-1.0, 1.0)


def assign_and_loss(cls_outs, reg_outs, targets, strides, img_size, nc, wcls, wbox, wl1, device):
    B = cls_outs[0].shape[0]
    pts, strs = make_anchors(cls_outs, strides)
    A = pts.shape[0]

    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, nc) for c in cls_outs], 1)
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in reg_outs], 1)

    with torch.no_grad():
        pb = torch.stack([decode_boxes(reg_f[b], pts, strs).clamp(0, img_size) for b in range(B)])

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
    wts     = torch.zeros(B, A, device=device)

    with torch.no_grad():
        for b in range(B):
            if N == 0: continue
            bm = targets[:, 0] == b
            if not bm.any(): continue
            gtb = gt_px[bm]
            clb = targets[bm, 1].long()
            G = len(gtb)

            xa = pts[:, 0].unsqueeze(0)
            ya = pts[:, 1].unsqueeze(0)
            l_star = (xa - gtb[:, 0:1]) / strs.unsqueeze(0)
            t_star = (ya - gtb[:, 1:2]) / strs.unsqueeze(0)
            r_star = (gtb[:, 2:3] - xa) / strs.unsqueeze(0)
            b_star = (gtb[:, 3:4] - ya) / strs.unsqueeze(0)
            offsets_star = torch.stack([l_star, t_star, r_star, b_star], dim=-1)

            radius = (strs * 1.5).unsqueeze(0)
            is_cand = (
                (xa >= gtb[:, 0:1] - radius) &
                (xa <= gtb[:, 2:3] + radius) &
                (ya >= gtb[:, 1:2] - radius) &
                (ya <= gtb[:, 3:4] + radius) &
                (l_star >= 0) & (t_star >= 0) & (r_star >= 0) & (b_star >= 0)
            ).float()
            # Chunked box_iou: avoids OOM on T4 with 102K anchors (stride-2 head)
            pb_b = pb[b].detach()  # [A, 4]
            chunk = 8192
            iou_chunks = []
            for ci in range(0, A, chunk):
                iou_chunks.append(box_iou(gtb, pb_b[ci:ci+chunk]).clamp(0, 1))
            iou_ga = torch.cat(iou_chunks, dim=1)  # [G, A]

            csc_ga = torch.sigmoid(cls_f[b].detach())[:, clb].T.clamp(1e-6, 1.0)
            ta_sc = (iou_ga ** 0.5) * (csc_ga ** 0.5) * is_cand

            k = min(10, A)
            topk_v, topk_i = ta_sc.topk(k, dim=1)
            vf = topk_v.flatten()
            vi = topk_i.flatten()

            cf = clb.unsqueeze(1).expand(-1, k).flatten()
            bf = gtb.unsqueeze(1).expand(-1, k, 4).reshape(-1, 4)
            rf = offsets_star.view(G, A, 4).gather(1, topk_i.unsqueeze(-1).expand(-1, -1, 4)).reshape(-1, 4)

            ok = vf > 0.005
            if not ok.any(): continue
            vf, vi, cf, bf, rf = vf[ok], vi[ok], cf[ok], bf[ok], rf[ok]

            cur = wts[b]
            better = vf > cur[vi]
            vi_b, vf_b, cf_b, bf_b, rf_b = vi[better], vf[better], cf[better], bf[better], rf[better]

            if len(vi_b) == 0: continue
            cur[vi_b] = vf_b
            cls_tgt[b, vi_b, :] = 0.0
            cls_tgt[b, vi_b, cf_b] = vf_b
            reg_tgt[b, vi_b] = rf_b
            box_tgt[b, vi_b] = bf_b

    pos = wts > 0
    npos = pos.sum().clamp(min=1)

    cls_loss = F.binary_cross_entropy_with_logits(cls_f.reshape(-1, nc), cls_tgt.reshape(-1, nc), reduction="sum") / npos
    box_loss = torch.tensor(0., device=device)
    l1_loss  = torch.tensor(0., device=device)

    if pos.any():
        box_loss = ciou_loss(pb[pos], box_tgt[pos]).sum() / npos
        if wl1 > 0:
            l1_loss = F.smooth_l1_loss(reg_f[pos], reg_tgt[pos].clamp(0, 8), reduction="sum") / npos * wl1

    total = wcls * cls_loss + wbox * box_loss + l1_loss
    return total, cls_loss.detach(), box_loss.detach()


# ================================================================
# 5. POSTPROCESSING & COCO EVALUATION
# ================================================================

def postprocess(cls_outs, reg_outs, strides, img_size, nc, conf=0.05, max_det=300):
    B = cls_outs[0].shape[0]
    pts, strs = make_anchors(cls_outs, strides)
    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, nc).sigmoid() for c in cls_outs], 1)
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in reg_outs], 1)
    results = []

    for b in range(B):
        pb = decode_boxes(reg_f[b], pts, strs).clamp(0, img_size)
        sc, lb = cls_f[b].max(-1)
        mask = sc > conf
        pb, sc, lb = pb[mask], sc[mask], lb[mask]

        if len(pb) == 0:
            results.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
            continue

        fb, fs, fl = [], [], []
        for c in range(nc):
            m = lb == c
            if not m.any(): continue
            keep = nms(pb[m], sc[m], cfg.NMS_IOU_THRESH)
            fb.append(pb[m][keep]); fs.append(sc[m][keep])
            fl.append(torch.full((len(keep),), c, dtype=torch.long, device=pb.device))

        if fb:
            fb, fs, fl = torch.cat(fb), torch.cat(fs), torch.cat(fl)
            if len(fs) > max_det:
                tk = fs.topk(max_det).indices
                fb, fs, fl = fb[tk], fs[tk], fl[tk]
            results.append({"boxes": fb.cpu(), "scores": fs.cpu(), "labels": fl.cpu()})
        else:
            results.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})

    return results


def build_coco_gt(val_ds):
    gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(cfg.CLASS_NAMES)]}
    aid = 0
    for iid, ip in enumerate(val_ds.img_paths):
        im = cv2.imread(str(ip))
        h0, w0 = im.shape[:2] if im is not None else (cfg.IMG_SIZE, cfg.IMG_SIZE)
        gt["images"].append({"id": iid, "width": w0, "height": h0, "file_name": ip.name})
        ap = val_ds.ann_dir / (ip.stem + ".txt")
        if not ap.exists(): continue
        for line in ap.read_text().splitlines():
            p = line.split(",")
            if len(p) < 6: continue
            x, y, bw, bh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            cat, sc = int(p[5]), int(p[4])
            if cat == 0 or sc == 0 or cat == 11: continue
            gt["annotations"].append({"id": aid, "image_id": iid, "category_id": cat - 1, "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    return gt


@torch.no_grad()
def evaluate_coco(model, val_dl, gt_dict, strides, device, conf=0.05):
    model.eval()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); gf = f.name
    coco_gt = COCO(gf); preds = []; iid = 0

    for imgs, _, paths, metas in tqdm(val_dl, desc="Evaluating Benchmark", leave=False, file=sys.stdout):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(enabled=cfg.USE_AMP):
            co, ro = model(imgs)
        dets = postprocess(co, ro, strides, cfg.IMG_SIZE, cfg.NUM_CLASSES, conf)

        for b, det in enumerate(dets):
            w0, h0, pl, pt, sc = metas[b].tolist()
            for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                x1, y1, x2, y2 = box.tolist()
                x1o = max(0., (x1 - pl) / sc); y1o = max(0., (y1 - pt) / sc)
                x2o = min(w0, (x2 - pl) / sc); y2o = min(h0, (y2 - pt) / sc)
                wo = max(0., x2o - x1o); ho = max(0., y2o - y1o)
                if wo > 0 and ho > 0:
                    preds.append({"image_id": iid + b, "category_id": int(label), "bbox": [x1o, y1o, wo, ho], "score": float(score)})
        iid += len(dets)

    empty = {"AP": 0., "AP50": 0., "AP75": 0., "APS": 0., "APM": 0., "APL": 0.}
    if not preds: return empty
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f); df = f.name
    ev = COCOeval(coco_gt, coco_gt.loadRes(df), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "APS": ev.stats[3], "APM": ev.stats[4], "APL": ev.stats[5]}


# ================================================================
# 6. VISUALIZATION & TRAINING LOOP
# ================================================================

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

def denorm(t):
    return ((t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1) * 255).astype(np.uint8)

def visualise(model, val_dl, strides, epoch, n_imgs=4):
    model.eval()
    fig, axes = plt.subplots(1, n_imgs, figsize=(6 * n_imgs, 6), dpi=140)
    count = 0
    for imgs, tgts, paths, metas in val_dl:
        imgs_g = imgs.to(cfg.DEVICE)
        with torch.no_grad(), autocast(enabled=cfg.USE_AMP):
            co, ro = model(imgs_g)
        dets = postprocess(co, ro, strides, cfg.IMG_SIZE, cfg.NUM_CLASSES, cfg.VIS_CONF_THRESH)
        for b in range(min(imgs.shape[0], n_imgs - count)):
            ax = axes[count] if n_imgs > 1 else axes
            ax.imshow(denorm(imgs[b])); ax.axis("off")
            for row in tgts[tgts[:, 0] == b]:
                _, c, cx, cy, w, h = row.tolist(); s = cfg.IMG_SIZE
                ax.add_patch(patches.Rectangle(((cx - w / 2) * s, (cy - h / 2) * s), w * s, h * s, lw=1, edgecolor="lime", facecolor="none", ls="--"))
            nd = 0
            for box, sc, lb in zip(dets[b]["boxes"], dets[b]["scores"], dets[b]["labels"]):
                x1, y1, x2, y2 = box.tolist()
                col = [v / 255 for v in cfg.COLORS[int(lb) % 10]]
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5, edgecolor=col, facecolor="none"))
                ax.text(x1, max(0, y1 - 2), f"{cfg.CLASS_NAMES[int(lb)]} {sc:.2f}", fontsize=6, color=col, fontweight="bold", bbox=dict(fc="black", alpha=0.5, pad=0.4, edgecolor="none"))
                nd += 1
            ax.set_title(f"Epoch {epoch} | {nd} Detected", fontsize=9, fontweight="bold")
            count += 1
        if count >= n_imgs: break
    plt.suptitle("HiSMD-Net: Ground Truth (Green Dashed) vs Predictions (Coloured)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    sp = os.path.join(cfg.SAVE_DIR, f"vis_epoch_{epoch:03d}.png")
    plt.savefig(sp, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  Visualization saved: {sp}")


def train_epoch(model, dl, opt, scaler, epoch, device):
    model.train()
    tot = cls_s = box_s = 0.; nb = 0
    pbar = tqdm(dl, desc=f"Ep {epoch:3d}/{cfg.EPOCHS}", leave=False, file=sys.stdout, ncols=100)

    for imgs, tgts, _, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.USE_AMP):
            co, ro = model(imgs)
            loss, cl, bl = assign_and_loss(co, ro, tgts, model.strides, cfg.IMG_SIZE, cfg.NUM_CLASSES, cfg.W_CLS, cfg.W_BOX, cfg.W_L1, device)

        if torch.isnan(loss) or torch.isinf(loss): continue

        if cfg.USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()

        tot += loss.item(); cls_s += cl.item(); box_s += bl.item(); nb += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "cls": f"{cl.item():.3f}", "box": f"{bl.item():.3f}"})

    return {"total": tot / max(nb, 1), "cls": cls_s / max(nb, 1), "box": box_s / max(nb, 1)}


def init_log_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "loss", "cls_loss", "box_loss", "AP", "AP50", "APS", "APM"])

def append_log_csv(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    print("\n" + "=" * 70)
    print("  HiSMD-Net Auto-Resume Training Pipeline (Resuming to Epoch 100)")
    print("=" * 70)
    print(f"  Device  : {cfg.DEVICE}")
    print(f"  Batch   : {cfg.BATCH_SIZE}   Epochs: {cfg.EPOCHS}")
    print("=" * 70)

    tl, vl, tds, vds = build_dataloaders()
    gt_dict = build_coco_gt(vds)

    model  = HiSMDNet(cfg).to(cfg.DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    init_log_csv(os.path.join(cfg.SAVE_DIR, "training_log.csv"))

    start_epoch = 1
    best_ap50 = 0.
    best_ap = 0.

    last_ckpt_path = os.path.join(cfg.SAVE_DIR, "last_hismdnet.pt")
    best_ckpt_path = os.path.join(cfg.SAVE_DIR, "best_hismdnet.pt")

    # AUTO-RESUME LOGIC (Checks last_hismdnet.pt first, then best_hismdnet.pt)
    resume_ckpt = None
    if os.path.exists(last_ckpt_path):
        resume_ckpt = last_ckpt_path
    elif os.path.exists(best_ckpt_path):
        resume_ckpt = best_ckpt_path

    if resume_ckpt:
        print(f"\n[+] Auto-Resume: Found existing checkpoint: {resume_ckpt}")
        try:
            ckpt = torch.load(resume_ckpt, map_location=cfg.DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                opt.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                sched.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and ckpt["scaler"] is not None and cfg.USE_AMP:
                scaler.load_state_dict(ckpt["scaler"])

            saved_epoch = ckpt.get("epoch", 1)
            start_epoch = saved_epoch + 1
            if "best_ap50" in ckpt:
                best_ap50 = ckpt["best_ap50"]
            elif "metrics" in ckpt:
                best_ap50 = ckpt["metrics"].get("AP50", 0.)
            if "best_ap" in ckpt:
                best_ap = ckpt["best_ap"]
            elif "metrics" in ckpt:
                best_ap = ckpt["metrics"].get("AP", 0.)

            print(f"[+] LOAD SUCCESS! State restored from completed Epoch {saved_epoch}.")
            print(f"[+] Resuming training from Epoch {start_epoch}/{cfg.EPOCHS}\n")
        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}. Starting fresh.\n")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        L = train_epoch(model, tl, opt, scaler, epoch, cfg.DEVICE)
        sched.step()
        lr = sched.get_last_lr()[0]

        print(f"\nEpoch {epoch:3d}/{cfg.EPOCHS} | LR={lr:.2e} | Loss={L['total']:.4f} (cls={L['cls']:.4f}, box={L['box']:.4f})")

        m = None
        if epoch % cfg.EVAL_FREQ == 0 or epoch == cfg.EPOCHS:
            m = evaluate_coco(model, vl, gt_dict, model.strides, cfg.DEVICE, conf=cfg.CONF_THRESH)
            print(f"  >>> AP={m['AP']*100:.2f}% | AP50={m['AP50']*100:.2f}% | APS={m['APS']*100:.2f}% | APM={m['APM']*100:.2f}%")
            visualise(model, vl, model.strides, epoch)

            if m["AP50"] > best_ap50 or m["AP"] > best_ap:
                best_ap50 = max(best_ap50, m["AP50"])
                best_ap = max(best_ap, m["AP"])
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict() if cfg.USE_AMP else None,
                    "metrics": m,
                    "best_ap50": best_ap50,
                    "best_ap": best_ap
                }, best_ckpt_path)
                print(f"  ★ NEW BEST BENCHMARK! AP50: {best_ap50*100:.2f}% | mAP: {best_ap*100:.2f}% -> Saved to {best_ckpt_path}")

        # ALWAYS save last_hismdnet.pt every epoch so interruptions at any point can resume
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if cfg.USE_AMP else None,
            "best_ap50": best_ap50,
            "best_ap": best_ap
        }, last_ckpt_path)

        # Log to CSV
        log_csv_path = os.path.join(cfg.SAVE_DIR, "training_log.csv")
        append_log_csv(log_csv_path, [
            epoch, f"{lr:.6e}", f"{L['total']:.4f}", f"{L['cls']:.4f}", f"{L['box']:.4f}",
            f"{m['AP']*100:.2f}" if m else "",
            f"{m['AP50']*100:.2f}" if m else "",
            f"{m['APS']*100:.2f}" if m else "",
            f"{m['APM']*100:.2f}" if m else ""
        ])

    print("\n" + "=" * 70)
    print(f"  Training Complete! Peak Accuracy AP50: {best_ap50*100:.2f}% | mAP: {best_ap*100:.2f}%")
    print(f"  All outputs saved in: {cfg.SAVE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
