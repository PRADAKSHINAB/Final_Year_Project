"""
research/test_image.py
======================
Interactive Single-Image Testing Script for Trained UAV Detection Models.

Features:
  1. If --image is provided, runs on that image directly.
  2. If --image is omitted, prompts with a GUI file dialog (Tkinter) if available,
     or asks via terminal prompt.
  3. Supports JPG, JPEG, PNG, WEBP, BMP.
  4. Configurable confidence threshold via --confidence (default: 0.50).
  5. Displays and logs:
     - Detected bounding boxes
     - Actual VisDrone class names
     - Confidence scores
     - Number of detections
     - Inference time
     - Device used (GPU/CPU)
  6. Saves annotated output to results/predictions/<image_name>_detected.jpg without
     overwriting the original image.
  7. If no objects meet the confidence threshold, cleanly reports: "No objects detected."

Usage:
  python research/test_image.py --checkpoint results/E4_full/checkpoints/best.pth
  python research/test_image.py --checkpoint results/E4_full/checkpoints/best.pth --image path/to/drone.jpg --confidence 0.4
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

VISDRONE_CLASSES = [
    "__background__",
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "pedestrian":      (255, 220,   0),
    "people":          (153, 211,  52),
    "bicycle":         ( 21, 204, 250),
    "car":             ( 94,  63, 244),
    "van":             (246, 130,  59),
    "truck":           (247,  85, 168),
    "tricycle":        (129, 185,  16),
    "awning-tricycle": ( 22, 115, 249),
    "bus":             ( 94, 197,  34),
    "motor":           (166, 184,  20),
}

EXPERIMENT_MAP = {
    "baseline":      {"use_attention": False, "use_sacm": False},
    "attention":     {"use_attention": True,  "use_sacm": False},
    "second_module": {"use_attention": False, "use_sacm": True},
    "final_model":   {"use_attention": True,  "use_sacm": True},
}


def _cls_name(idx: int) -> str:
    return VISDRONE_CLASSES[idx] if 0 <= idx < len(VISDRONE_CLASSES) else f"cls_{idx}"


def select_image_file() -> Optional[str]:
    """Open GUI file dialog or prompt in terminal."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select UAV / Drone Image for Testing",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
                ("All files", "*.*"),
            ]
        )
        root.destroy()
        if file_path:
            return file_path
    except Exception:
        pass

    # Terminal fallback
    print("\n[PROMPT] Enter the path to the image file:")
    try:
        user_input = input("Image path: ").strip().strip('"').strip("'")
        if user_input and Path(user_input).exists():
            return user_input
    except (EOFError, KeyboardInterrupt):
        return None
    return None


def draw_detections(
    image_bgr: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
) -> np.ndarray:
    out = image_bgr.copy()
    for box, lbl, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        name = _cls_name(lbl)
        color = _COLORS_BGR.get(name, (200, 200, 200))

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label tag
        tag = f"{name} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 6)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + bl - 1), color, -1)
        cv2.putText(out, tag, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def run_interactive_inference(
    checkpoint_path: str,
    image_path: Optional[str] = None,
    confidence_thresh: float = 0.50,
    experiment: str = "final_model",
    img_size: int = 1280,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> None:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint file not found: {ckpt_file}")
        sys.exit(1)

    if not image_path:
        print("[INFO] No image path provided. Opening file selector...")
        image_path = select_image_file()
        if not image_path:
            print("[INFO] No image selected. Exiting.")
            return

    img_p = Path(image_path)
    if not img_p.exists():
        print(f"[ERROR] Image path does not exist: {img_p}")
        sys.exit(1)

    # 1. Device detection
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n{'='*60}")
    print("UAV SMALL-OBJECT DETECTION — INTERACTIVE TEST")
    print(f"{'='*60}")
    print(f"Model Checkpoint : {ckpt_file.resolve()}")
    print(f"Target Image     : {img_p.resolve()}")
    print(f"Device           : {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    print(f"Confidence Thresh: {confidence_thresh:.2f}")

    # 2. Build model architecture
    flags = EXPERIMENT_MAP.get(experiment, EXPERIMENT_MAP["final_model"])
    model, model_name = build_proposed_model(
        use_attention=flags["use_attention"],
        use_sacm=flags["use_sacm"],
        pretrained_backbone=False,
    )

    # 3. Load checkpoint
    ckpt = torch.load(str(ckpt_file), map_location=dev)
    state = ckpt.get("model", ckpt)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    model = model.to(dev).eval()

    # 4. Read image
    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        print(f"[ERROR] Failed to decode image: {img_p}")
        sys.exit(1)
    orig_h, orig_w = img_bgr.shape[:2]

    # 5. Preprocess
    img_resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(dev)

    # 6. Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model([tensor])[0]
    dt_ms = (time.perf_counter() - t0) * 1000

    # 7. Post-process & filter
    keep = out["scores"].cpu() >= confidence_thresh
    boxes = out["boxes"].cpu()[keep].numpy().tolist()
    labels = out["labels"].cpu()[keep].numpy().tolist()
    scores = out["scores"].cpu()[keep].numpy().tolist()

    # Rescale back to original image dimensions
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    boxes_orig = [
        [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
        for b in boxes
    ]

    # 8. Report results
    print(f"\nInference Time   : {dt_ms:.1f} ms")
    print(f"Total Detections : {len(boxes_orig)}")
    print("-" * 60)

    if not boxes_orig:
        print("No objects detected above confidence threshold.")
    else:
        print(f"{'Class Name':<18} {'Confidence':<12} {'Bounding Box [x1, y1, x2, y2]'}")
        print("-" * 60)
        for box, lbl, score in sorted(zip(boxes_orig, labels, scores), key=lambda x: -x[2]):
            name = _cls_name(lbl)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            print(f"{name:<18} {score:<12.4f} [{x1}, {y1}, {x2}, {y2}]")

    # 9. Save result image
    if output_path is None:
        out_dir = PROJECT_ROOT / "results" / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{img_p.stem}_detected.jpg"
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    annotated = draw_detections(img_bgr, boxes_orig, labels, scores)
    cv2.imwrite(str(out_file), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nSaved Annotated Result Image: {out_file.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Single-Image UAV Object Detection Tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", "--model", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, file dialog opens.")
    parser.add_argument("--confidence", "--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--experiment", default="final_model", choices=list(EXPERIMENT_MAP), help="Model architecture")
    parser.add_argument("--img_size", type=int, default=1280, help="Inference resolution")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output", default=None, help="Custom output image path")
    args = parser.parse_args()

    run_interactive_inference(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        confidence_thresh=args.confidence,
        experiment=args.experiment,
        img_size=args.img_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
