"""
research/inference.py
=====================
Single-image and multi-image batch inference for SOA-FasterRCNN experiments.

Usage — single image:
    python research/inference.py \
        --model experiments/final_model/weights/best_model.pth \
        --experiment final_model \
        --image path/to/image.jpg \
        --output results/test_prediction.jpg

Usage — multi-image directory:
    python research/inference.py \
        --model experiments/final_model/weights/best_model.pth \
        --experiment final_model \
        --input_dir test_images/ \
        --output_dir results/test_predictions/

Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.models.proposed_model import build_proposed_model

# ──────────────────────────────────────────────────────────────────────────────
VISDRONE_CLASSES = [
    "__background__",
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

_COLORS_BGR: Dict[str, Tuple[int,int,int]] = {
    "pedestrian":      (255,220,  0), "people":     (153,211, 52),
    "bicycle":         ( 21,204,250), "car":        ( 94, 63,244),
    "van":             (246,130, 59), "truck":      (247, 85,168),
    "tricycle":        (129,185, 16), "awning-tricycle":( 22,115,249),
    "bus":             ( 94,197, 34), "motor":      (166,184, 20),
}

EXPERIMENT_MAP = {
    "baseline":      {"use_attention": False, "use_sacm": False},
    "attention":     {"use_attention": True,  "use_sacm": False},
    "second_module": {"use_attention": False, "use_sacm": True},
    "final_model":   {"use_attention": True,  "use_sacm": True},
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _cls_name(idx: int) -> str:
    return VISDRONE_CLASSES[idx] if 0 <= idx < len(VISDRONE_CLASSES) else f"cls_{idx}"


def _draw(
    img_bgr: np.ndarray,
    boxes: List,
    labels: List[int],
    scores: List[float],
) -> np.ndarray:
    out = img_bgr.copy()
    for box, lbl, score in zip(boxes, labels, scores):
        x1,y1,x2,y2 = (int(v) for v in box)
        name  = _cls_name(lbl)
        color = _COLORS_BGR.get(name, (200,200,200))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw,th),bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1-4, th+6)
        cv2.rectangle(out, (x1,ty-th-4),(x1+tw+4,ty+bl-1), color, -1)
        cv2.putText(out, text, (x1+2,ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, experiment: str = "baseline", device: str = "cuda"):
    """Load model from checkpoint."""
    flags = EXPERIMENT_MAP.get(experiment, EXPERIMENT_MAP["baseline"])
    model, model_name = build_proposed_model(
        use_attention=flags["use_attention"],
        use_sacm=flags["use_sacm"],
        pretrained_backbone=False,
    )
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(checkpoint, map_location=dev)
    state = ckpt.get("model", ckpt)
    own   = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    epoch = ckpt.get("epoch", "?")
    map50 = ckpt.get("metrics", {}).get("map50", "?")
    print(f"[LOADED] {model_name}  epoch={epoch}  mAP@50={map50}")
    print(f"         {len(matched)}/{len(own)} parameters matched")
    return model.to(dev).eval(), model_name, dev


@torch.inference_mode()
def infer_single(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    img_size: int = 1280,
    conf: float = 0.30,
    save_json: bool = True,
) -> List[Dict]:
    """Run inference on one image; save result image and prints detections."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    # Pre-process
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
    tensor  = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0).to(device)

    t0  = time.perf_counter()
    out = model([tensor])[0]
    ms  = (time.perf_counter() - t0) * 1000

    keep   = out["scores"].cpu() >= conf
    boxes  = out["boxes"].cpu()[keep].numpy().tolist()
    labels = out["labels"].cpu()[keep].numpy().tolist()
    scores = out["scores"].cpu()[keep].numpy().tolist()

    # Scale boxes back to original resolution
    sx, sy = orig_w / img_size, orig_h / img_size
    boxes_orig = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes]

    result_img = _draw(img_bgr, boxes_orig, labels, scores)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    detections = []
    print(f"\nDetected ({len(boxes)} objects, {ms:.1f} ms):")
    for box, lbl, score in sorted(zip(boxes_orig, labels, scores), key=lambda x: -x[2]):
        name = _cls_name(lbl)
        x1,y1,x2,y2 = box
        print(f"  {name}: {score:.4f}   [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
        detections.append({
            "class_name": name, "class_id": lbl, "confidence": round(score, 6),
            "x1": round(x1,1), "y1": round(y1,1), "x2": round(x2,1), "y2": round(y2,1),
            "width": round(x2-x1,1), "height": round(y2-y1,1),
        })

    print(f"\nSaved output image: {out_path}")

    if save_json:
        jp = out_path.with_suffix(".json")
        with open(jp, "w") as f:
            json.dump({"image_name": Path(image_path).name, "inference_ms": ms,
                       "detections": detections}, f, indent=2)
        print(f"Saved detections JSON: {jp}")

    return detections


@torch.inference_mode()
def infer_batch(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    img_size: int = 1280,
    conf: float = 0.30,
    save_json: bool = True,
) -> None:
    """Process all images in a directory and save annotated images + JSON."""
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[WARNING] No images found in {input_dir}")
        return

    print(f"[BATCH] Processing {len(images)} images -> {output_dir}")
    all_dets, total_ms = [], 0.0

    for i, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
        tensor  = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0).to(device)

        t0  = time.perf_counter()
        out = model([tensor])[0]
        ms  = (time.perf_counter() - t0) * 1000
        total_ms += ms

        keep   = out["scores"].cpu() >= conf
        boxes  = out["boxes"].cpu()[keep].numpy().tolist()
        labels = out["labels"].cpu()[keep].numpy().tolist()
        scores = out["scores"].cpu()[keep].numpy().tolist()
        sx, sy = orig_w/img_size, orig_h/img_size
        boxes_orig = [[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy] for b in boxes]

        result_img = _draw(img_bgr, boxes_orig, labels, scores)
        out_path   = output_dir / img_path.name
        cv2.imwrite(str(out_path), result_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        dets = []
        for box, lbl, score in zip(boxes_orig, labels, scores):
            x1,y1,x2,y2 = box
            dets.append({
                "class_name": _cls_name(lbl), "class_id": lbl,
                "confidence": round(score, 6),
                "x1": round(x1,1), "y1": round(y1,1),
                "x2": round(x2,1), "y2": round(y2,1),
                "width": round(x2-x1,1), "height": round(y2-y1,1),
            })
        
        all_dets.append({
            "image_name": img_path.name,
            "detections": dets
        })

        print(f"  [{i:4d}/{len(images)}]  {img_path.name:<30}  "
              f"{len(boxes):3d} dets  {ms:.0f}ms")

    avg_ms = total_ms / len(images) if images else 0
    fps    = 1000 / avg_ms if avg_ms else 0
    print(f"\n[DONE]  avg={avg_ms:.1f}ms/img  FPS={fps:.1f}")
    print(f"        results -> {output_dir}")

    if save_json:
        jp = output_dir / "predictions.json"
        with open(jp, "w") as f:
            json.dump(all_dets, f, indent=2)
        print(f"        JSON   -> {jp}")


# ──────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="SOA-FasterRCNN inference script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Accept both --model and --checkpoint for compatibility
    p.add_argument("--model", "--checkpoint", dest="checkpoint", required=True,
                   help="Path to model weights checkpoint (.pth)")
    p.add_argument("--experiment", default="baseline",
                   choices=list(EXPERIMENT_MAP),
                   help="Experiment architecture variant")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",     help="Single image path")
    mode.add_argument("--input_dir", help="Directory of images for batch inference")

    p.add_argument("--output",     default=None, help="Output path (single image)")
    p.add_argument("--output_dir", default=None, help="Output dir (batch)")

    p.add_argument("--conf",     type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--img_size", type=int,   default=1280, help="Image size")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_json",  action="store_true")
    return p.parse_args()


def main():
    args  = _parse()
    model, model_name, dev = load_model(args.checkpoint, args.experiment, args.device)
    print(f"[INFERENCE] conf≥{args.conf}  size={args.img_size}  device={dev}\n")

    if args.image:
        out = args.output or str(
            Path(args.image).with_name(f"result_{Path(args.image).stem}.jpg"))
        infer_single(model, args.image, out, dev,
                     img_size=args.img_size, conf=args.conf, save_json=not args.no_json)
    else:
        out_dir = args.output_dir or str(Path(args.input_dir) / "predictions")
        infer_batch(model, args.input_dir, out_dir, dev,
                    img_size=args.img_size, conf=args.conf, save_json=not args.no_json)


if __name__ == "__main__":
    main()
