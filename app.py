# ================================================================
# HiSMD-Net Web Application Backend
# FastAPI server that loads best_hismdnet.pt and serves inference
# Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# ================================================================

import os, sys, json, math, io, base64, time, random, subprocess
from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")

# ── User package path for HPC / restricted environments ─────────
MY_PACKAGES = os.path.expanduser("~/my_python_packages")
os.makedirs(MY_PACKAGES, exist_ok=True)
if MY_PACKAGES not in sys.path:
    sys.path.insert(0, MY_PACKAGES)

def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", MY_PACKAGES, "--upgrade", pip_name])
        if MY_PACKAGES not in sys.path:
            sys.path.insert(0, MY_PACKAGES)

ensure_package("fastapi", "fastapi")
ensure_package("uvicorn", "uvicorn")
ensure_package("multipart", "python-multipart")

# ── FastAPI ─────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── ML stack ────────────────────────────────────────────────────
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou
import torchvision.models as models

# ================================================================
# 1.  EXACT HiSMD-Net Architecture  (must match train.py)
# ================================================================

class ASRFR(nn.Module):
    """Adaptive Sub-pixel Resolution Feature Reconstructor"""
    def __init__(self, hidden=32):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_shuffle   = nn.PixelShuffle(2)
        self.sr_net = nn.Sequential(
            nn.Conv2d(12, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 12, 3, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        return x + self.alpha.clamp(0., 1.) * self.pixel_shuffle(self.sr_net(self.pixel_unshuffle(x)))


class SpatialSSMBlock(nn.Module):
    """Spatial State-Space Mamba-style attention block"""
    def __init__(self, dim):
        super().__init__()
        self.dw    = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        self.norm  = nn.BatchNorm2d(dim)
        self.pw1   = nn.Conv2d(dim, dim * 2, 1)
        self.act   = nn.GELU()
        self.pw2   = nn.Conv2d(dim * 2, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class HiSMDNet(nn.Module):
    """Hierarchical Super-resolved Mamba Detector — 16.49M params"""
    def __init__(self):
        super().__init__()
        self.asrfr = ASRFR(32)
        base = models.resnet50(weights=None)          # load weights from checkpoint
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stage1 = base.layer1                     # stride-4
        self.stage2 = nn.Sequential(base.layer2, SpatialSSMBlock(512))   # stride-8
        self.stage3 = nn.Sequential(base.layer3, SpatialSSMBlock(1024))  # stride-16
        nc = 128
        self.lat3 = nn.Conv2d(1024, nc, 1)
        self.lat2 = nn.Conv2d(512,  nc, 1)
        self.lat1 = nn.Conv2d(256,  nc, 1)
        self.fuse3 = nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1), nn.BatchNorm2d(nc), nn.SiLU(inplace=True))
        self.fuse2 = nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1), nn.BatchNorm2d(nc), nn.SiLU(inplace=True))
        self.fuse1 = nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1), nn.BatchNorm2d(nc), nn.SiLU(inplace=True))
        self.hires_head = nn.Sequential(
            nn.Conv2d(nc, nc * 4, 3, 1, 1), nn.PixelShuffle(2),
            nn.BatchNorm2d(nc), nn.SiLU(inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1), nn.BatchNorm2d(nc), nn.SiLU(inplace=True))
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1), nn.SiLU(inplace=True), nn.Conv2d(nc, 10, 1))
            for _ in range(4)])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1), nn.SiLU(inplace=True), nn.Conv2d(nc, 4, 1))
            for _ in range(4)])
        self.strides = [2, 4, 8, 16]
        prior = 0.01
        for h in self.cls_heads: nn.init.constant_(h[-1].bias, -math.log((1 - prior) / prior))
        for h in self.reg_heads: nn.init.constant_(h[-1].bias, 0.5)

    def forward(self, x):
        x  = self.asrfr(x)
        s  = self.stem(x)
        c2 = self.stage1(s)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        p4 = self.lat3(c4)
        p3 = self.fuse3(self.lat2(c3) + F.interpolate(p4, scale_factor=2, mode="nearest"))
        p2 = self.fuse2(self.lat1(c2) + F.interpolate(p3, scale_factor=2, mode="nearest"))
        p1 = self.hires_head(p2)
        feats    = [p1, p2, p3, p4]
        cls_outs = [self.cls_heads[i](f) for i, f in enumerate(feats)]
        reg_outs = [F.softplus(self.reg_heads[i](f)).clamp(max=8.) for i, f in enumerate(feats)]
        return cls_outs, reg_outs


# ================================================================
# 2.  DECODING & POSTPROCESSING
# ================================================================

def make_anchors(cls_outs, strides, device):
    pts, strs = [], []
    for c, s in zip(cls_outs, strides):
        H, W = c.shape[-2:]
        gy, gx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing="ij")
        pts.append(torch.stack([(gx.flatten() + .5) * s, (gy.flatten() + .5) * s], 1))
        strs.append(torch.full((H * W,), s, device=device, dtype=torch.float32))
    return torch.cat(pts), torch.cat(strs)


def decode_boxes(reg, pts, strs):
    off = reg * strs.unsqueeze(1)
    return torch.stack([pts[:, 0] - off[:, 0], pts[:, 1] - off[:, 1],
                        pts[:, 0] + off[:, 2], pts[:, 1] + off[:, 3]], dim=1)


@torch.no_grad()
def postprocess(cls_outs, reg_outs, strides, img_size, nc, device,
                conf=0.05, iou_thr=0.50, max_det=300):
    pts, strs = make_anchors(cls_outs, strides, device)
    cls_f = torch.cat([c.permute(0, 2, 3, 1).reshape(1, -1, nc).sigmoid() for c in cls_outs], 1)[0]
    reg_f = torch.cat([r.permute(0, 2, 3, 1).reshape(1, -1, 4) for r in reg_outs], 1)[0]
    pb    = decode_boxes(reg_f, pts, strs).clamp(0, img_size)
    sc, lb = cls_f.max(-1)
    mask   = sc > conf
    pb, sc, lb = pb[mask], sc[mask], lb[mask]
    if len(pb) == 0:
        return [], [], []
    # per-class NMS
    fb, fs, fl = [], [], []
    for c in range(nc):
        m = lb == c
        if not m.any(): continue
        keep = nms(pb[m], sc[m], iou_thr)
        fb.append(pb[m][keep]); fs.append(sc[m][keep])
        fl.append(torch.full((len(keep),), c, dtype=torch.long, device=device))
    if not fb:
        return [], [], []
    fb, fs, fl = torch.cat(fb), torch.cat(fs), torch.cat(fl)
    if len(fs) > max_det:
        tk = fs.topk(max_det).indices
        fb, fs, fl = fb[tk], fs[tk], fl[tk]
    return fb.cpu().numpy().tolist(), fs.cpu().numpy().tolist(), fl.cpu().numpy().tolist()


# ================================================================
# 3.  IMAGE PREPROCESSING  (exact letterbox from train.py)
# ================================================================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 640

def letterbox(img_bgr):
    """BGR numpy → tensor [1,3,640,640], also returns (pad_l, pad_t, scale)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    scale   = IMG_SIZE / max(h, w)
    nw, nh  = int(w * scale), int(h * scale)
    img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_l   = (IMG_SIZE - nw) // 2
    pad_t   = (IMG_SIZE - nh) // 2
    pad_r   = IMG_SIZE - nw - pad_l
    pad_b   = IMG_SIZE - nh - pad_t
    img_rgb = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor  = ((img_rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(tensor).unsqueeze(0), pad_l, pad_t, scale


def undo_letterbox(box, pad_l, pad_t, scale, orig_w, orig_h):
    """Map predicted pixel box back to original image coordinates."""
    x1, y1, x2, y2 = box
    x1o = max(0., (x1 - pad_l) / scale)
    y1o = max(0., (y1 - pad_t) / scale)
    x2o = min(orig_w, (x2 - pad_l) / scale)
    y2o = min(orig_h, (y2 - pad_t) / scale)
    return [round(x1o), round(y1o), round(x2o), round(y2o)]


# ================================================================
# 4.  CONSTANTS
# ================================================================

CLASS_NAMES  = ["pedestrian", "people", "bicycle", "car", "van",
                "truck", "tricycle", "awning-tricycle", "bus", "motor"]

CLASS_COLORS_BGR = [
    (71, 99, 255),   # pedestrian  – red
    (50, 144, 255),  # people      – orange
    (32, 165, 218),  # bicycle     – gold
    (255, 185, 15),  # car         – cyan
    (235, 82, 111),  # van         – purple
    (200, 100, 255), # truck       – pink
    (86, 180, 80),   # tricycle    – green
    (255, 130, 30),  # awning-tri  – blue
    (30, 200, 255),  # bus         – amber
    (130, 130, 130), # motor       – slate
]

BENCHMARK_DATA = {
    "models": ["HiSMD-Net\n(Ours)", "UAVDet\n(Base Paper)", "YOLOv9-c", "YOLOv8-m", "RT-DETR-R50", "Faster R-CNN"],
    "ap50":   [68.43, 64.20, 62.40, 60.80, 59.10, 47.20],
    "map":    [47.80, 43.10, 41.60, 39.20, 38.90, 28.70],
    "aps":    [30.23, 24.50, 18.10, 17.40, 14.20, 9.10],
}

CHECKPOINT_PATHS = [
    "./hismdet_output/best_hismdnet.pt",
    "/content/hismdet_output/best_hismdnet.pt",
    "/content/drive/MyDrive/hismdet_output/best_hismdnet.pt",
    str(Path.home() / "hismdet_output/best_hismdnet.pt"),
]


# ================================================================
# 5.  MODEL INITIALIZATION
# ================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = None
CKPT_PATH_USED = None

def load_model():
    global MODEL, CKPT_PATH_USED
    model = HiSMDNet().to(DEVICE)
    for ckpt_path in CHECKPOINT_PATHS:
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                state = ckpt.get("model", ckpt)
                model.load_state_dict(state, strict=True)
                model.eval()
                MODEL = model
                CKPT_PATH_USED = ckpt_path
                vram_str = ""
                if DEVICE == "cuda":
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    vram_str = f" | VRAM: {vram:.1f}GB"
                print(f"[✓] HiSMD-Net loaded from: {ckpt_path}")
                print(f"[✓] Device: {DEVICE.upper()}{vram_str}")
                return True
            except Exception as e:
                print(f"[!] Failed to load {ckpt_path}: {e}")
    # Demo mode — random weights for UI preview
    model.eval()
    MODEL = model
    CKPT_PATH_USED = "demo_mode"
    print("[!] No checkpoint found. Running in DEMO MODE with random weights.")
    return False

load_model()


# ================================================================
# 6.  ANNOTATION DRAWING
# ================================================================

def draw_detections(img_bgr, boxes, scores, labels,
                    class_filter: Optional[List[int]] = None,
                    show_conf: bool = True,
                    line_thickness: int = 2):
    """Draw bounding boxes with class labels and confidence on BGR image."""
    overlay = img_bgr.copy()
    h, w    = img_bgr.shape[:2]
    font    = cv2.FONT_HERSHEY_SIMPLEX

    for box, score, label in zip(boxes, scores, labels):
        lbl = int(label)
        if class_filter is not None and lbl not in class_filter:
            continue
        color   = CLASS_COLORS_BGR[lbl % len(CLASS_COLORS_BGR)]
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Filled rectangle overlay for large boxes, thin for small objects
        area = (x2 - x1) * (y2 - y1)
        alpha = 0.18 if area > 1000 else 0.08
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Border
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, line_thickness)

        # Label background
        txt   = f"{CLASS_NAMES[lbl]} {score:.2f}" if show_conf else CLASS_NAMES[lbl]
        fs    = max(0.35, min(0.65, (x2 - x1) / 120))
        thick = 1
        (tw, th), _ = cv2.getTextSize(txt, font, fs, thick)
        ty    = y1 - 4 if y1 - th - 8 >= 0 else y2 + th + 4
        cv2.rectangle(img_bgr, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(img_bgr, txt, (x1 + 3, ty), font, fs, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
    return img_bgr


def img_to_b64(img_bgr) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ================================================================
# 7.  SAMPLE IMAGES
# ================================================================

SAMPLE_DIRS = [
    "./sample_images",
    str(Path.home() / "sample_images"),
    "./Dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/images",
    "./Dataset/VisDrone2019-DET-val/images",
    str(Path.home() / "Dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/images"),
    str(Path.home() / "Dataset/VisDrone2019-DET-val/images"),
    str(Path.home() / "uavdet_system/uavdet-system/Dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/images"),
    "/content/Dataset/VisDrone2019-DET-val/VisDrone2019-DET-val/images",
]

_SAMPLE_CACHE: List[dict] = []

def load_sample_images(n=6):
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE:
        return _SAMPLE_CACHE
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = []
    for d in SAMPLE_DIRS:
        p = Path(d)
        if p.exists():
            imgs = [f for f in sorted(p.iterdir()) if f.suffix.lower() in exts]
            if imgs:
                break
    if not imgs:
        return []
    # Pick diverse samples (spread across sorted list)
    indices = [int(len(imgs) * i / n) for i in range(n)]
    result  = []
    for i, idx in enumerate(indices):
        f = imgs[idx]
        img = cv2.imread(str(f))
        if img is None:
            continue
        # Thumbnail for preview
        thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
        result.append({
            "id":        i,
            "name":      f"Scene #{i+1} ({f.name})",
            "path":      str(f),
            "thumb_b64": img_to_b64(thumb),
        })
    _SAMPLE_CACHE = result
    return result


# ================================================================
# 8.  FASTAPI APPLICATION
# ================================================================

app = FastAPI(title="HiSMD-Net Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/api/status")
async def status():
    device_info = DEVICE.upper()
    vram_gb     = None
    gpu_name    = None
    if DEVICE == "cuda":
        props    = torch.cuda.get_device_properties(0)
        vram_gb  = round(props.total_memory / (1024**3), 1)
        gpu_name = props.name
    return {
        "ready":      MODEL is not None,
        "checkpoint": CKPT_PATH_USED,
        "device":     device_info,
        "gpu":        gpu_name,
        "vram_gb":    vram_gb,
        "metrics": {
            "AP50": 68.43,
            "mAP":  47.80,
            "APS":  30.23,
            "APM":  69.72,
            "APL":  75.00,
        }
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou:  float = Query(0.50, ge=0.10, le=0.90),
    classes: Optional[str] = Query(None),   # comma-separated class ids, e.g. "0,1,3"
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read image
    raw   = await file.read()
    arr   = np.frombuffer(raw, np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    orig_h, orig_w = img.shape[:2]

    # Class filter
    class_filter = None
    if classes:
        try:
            class_filter = [int(c.strip()) for c in classes.split(",") if c.strip().isdigit()]
        except:
            pass

    # Inference
    tensor, pad_l, pad_t, scale = letterbox(img)
    tensor = tensor.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        cls_outs, reg_outs = MODEL(tensor)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    boxes, scores, labels = postprocess(
        cls_outs, reg_outs, MODEL.strides, IMG_SIZE, 10, DEVICE, conf, iou)

    # Map boxes to original image coordinates
    orig_boxes = []
    det_list   = []
    class_counts = {}

    for box, score, label in zip(boxes, scores, labels):
        lbl = int(label)
        if class_filter is not None and lbl not in class_filter:
            continue
        orig_box = undo_letterbox(box, pad_l, pad_t, scale, orig_w, orig_h)
        orig_boxes.append(orig_box)
        cls_name = CLASS_NAMES[lbl]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        # Object size category (COCO standard)
        bw = orig_box[2] - orig_box[0]; bh = orig_box[3] - orig_box[1]
        area = bw * bh
        size_cat = "small" if area < 1024 else ("medium" if area < 9216 else "large")
        det_list.append({
            "class_id":   lbl,
            "class_name": cls_name,
            "confidence": round(float(score), 4),
            "box":        orig_box,      # [x1,y1,x2,y2] in original image pixels
            "box_norm":   [round(orig_box[0]/orig_w,4), round(orig_box[1]/orig_h,4),
                           round(orig_box[2]/orig_w,4), round(orig_box[3]/orig_h,4)],
            "size_cat":   size_cat,
        })

    # Draw on original image
    annotated = img.copy()
    if det_list:
        draw_detections(annotated, orig_boxes,
                        [d["confidence"] for d in det_list],
                        [d["class_id"]   for d in det_list],
                        class_filter=None)

    return {
        "total_detections": len(det_list),
        "class_counts":     class_counts,
        "detections":       det_list,
        "latency_ms":       latency_ms,
        "image_size":       [orig_w, orig_h],
        "annotated_image":  img_to_b64(annotated),
    }


@app.post("/api/predict_sample/{sample_id}")
async def predict_sample(
    sample_id: int,
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou:  float = Query(0.45, ge=0.10, le=0.90),
):
    """Run detection on a pre-loaded sample image by ID."""
    samples = load_sample_images()
    if sample_id >= len(samples):
        raise HTTPException(status_code=404, detail="Sample not found")
    path = samples[sample_id]["path"]
    img  = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=500, detail="Cannot read sample image")

    # Wrap as UploadFile-like using a temp file approach
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    orig_h, orig_w = img.shape[:2]

    tensor, pad_l, pad_t, scale = letterbox(img)
    tensor = tensor.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        cls_outs, reg_outs = MODEL(tensor)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    boxes, scores, labels = postprocess(
        cls_outs, reg_outs, MODEL.strides, IMG_SIZE, 10, DEVICE, conf, iou)

    orig_boxes = []; det_list = []; class_counts = {}
    for box, score, label in zip(boxes, scores, labels):
        lbl = int(label)
        orig_box = undo_letterbox(box, pad_l, pad_t, scale, orig_w, orig_h)
        orig_boxes.append(orig_box)
        cls_name = CLASS_NAMES[lbl]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        bw = orig_box[2] - orig_box[0]; bh = orig_box[3] - orig_box[1]
        area = bw * bh
        size_cat = "small" if area < 1024 else ("medium" if area < 9216 else "large")
        det_list.append({
            "class_id": lbl, "class_name": cls_name,
            "confidence": round(float(score), 4), "box": orig_box,
            "box_norm": [round(orig_box[0]/orig_w,4), round(orig_box[1]/orig_h,4),
                         round(orig_box[2]/orig_w,4), round(orig_box[3]/orig_h,4)],
            "size_cat": size_cat,
        })

    annotated = img.copy()
    if det_list:
        draw_detections(annotated, orig_boxes,
                        [d["confidence"] for d in det_list],
                        [d["class_id"]   for d in det_list])
    return {
        "total_detections": len(det_list), "class_counts": class_counts,
        "detections": det_list, "latency_ms": latency_ms,
        "image_size": [orig_w, orig_h], "annotated_image": img_to_b64(annotated),
        "original_image": img_to_b64(img),
    }


@app.get("/api/samples")
async def get_samples():
    return {"samples": load_sample_images()}


@app.get("/api/benchmark")
async def get_benchmark():
    return BENCHMARK_DATA


# ================================================================
# 9.  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import webbrowser, threading
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
