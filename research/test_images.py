"""
research/test_images.py
=======================
Multiple-Image Directory Batch Testing Script for Trained UAV Detection Models.

Features:
  1. Accepts an input folder containing images (--input_dir or terminal/GUI prompt).
  2. Processes all supported images (.jpg, .jpeg, .png, .webp, .bmp, .tiff).
  3. Draws detection bounding boxes, actual VisDrone class names, and confidence scores.
  4. Saves each output to results/predictions/<image_name>_detected.jpg (configurable --output_dir).
  5. Generates a structured results/predictions/batch_predictions.json with all detection records.
  6. Reports progress, detection counts, and average frames per second (FPS).

Usage:
  python research/test_images.py --checkpoint results/E4_full/checkpoints/best.pth --input_dir path/to/drone_images/
  python research/test_images.py --checkpoint results/E4_full/checkpoints/best.pth --confidence 0.4
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
from research.test_image import _cls_name, draw_detections, EXPERIMENT_MAP

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def select_folder() -> Optional[str]:
    """Open GUI folder selector or prompt in terminal."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Folder of UAV / Drone Images")
        root.destroy()
        if folder:
            return folder
    except Exception:
        pass

    print("\n[PROMPT] Enter the path to the directory containing images:")
    try:
        user_input = input("Input Directory: ").strip().strip('"').strip("'")
        if user_input and Path(user_input).is_dir():
            return user_input
    except (EOFError, KeyboardInterrupt):
        return None
    return None


def run_batch_folder_inference(
    checkpoint_path: str,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    confidence_thresh: float = 0.50,
    experiment: str = "final_model",
    img_size: int = 1280,
    device: str = "cuda",
) -> None:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_file}")
        sys.exit(1)

    if not input_dir:
        input_dir = select_folder()
        if not input_dir:
            print("[INFO] No folder selected. Exiting.")
            return

    in_path = Path(input_dir)
    if not in_path.is_dir():
        print(f"[ERROR] Input path is not a directory: {in_path}")
        sys.exit(1)

    if output_dir is None:
        out_path = PROJECT_ROOT / "results" / "predictions"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_files:
        print(f"[WARNING] No supported images found in {in_path}")
        return

    # Device
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n{'='*60}")
    print("UAV SMALL-OBJECT DETECTION — BATCH FOLDER INFERENCE")
    print(f"{'='*60}")
    print(f"Checkpoint       : {ckpt_file.resolve()}")
    print(f"Input Directory  : {in_path.resolve()} ({len(image_files)} images)")
    print(f"Output Directory : {out_path.resolve()}")
    print(f"Device           : {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    print(f"Confidence Thresh: {confidence_thresh:.2f}")

    # Build model
    flags = EXPERIMENT_MAP.get(experiment, EXPERIMENT_MAP["final_model"])
    model, model_name = build_proposed_model(
        use_attention=flags["use_attention"],
        use_sacm=flags["use_sacm"],
        pretrained_backbone=False,
    )

    # Load weights
    ckpt = torch.load(str(ckpt_file), map_location=dev)
    state = ckpt.get("model", ckpt)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    model = model.to(dev).eval()

    all_results = []
    total_time_ms = 0.0
    total_detections = 0

    print("\nProcessing images...")
    for idx, img_p in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"  [{idx:3d}/{len(image_files)}] SKIP (cannot decode): {img_p.name}")
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        img_resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model([tensor])[0]
        dt_ms = (time.perf_counter() - t0) * 1000
        total_time_ms += dt_ms

        keep = out["scores"].cpu() >= confidence_thresh
        boxes = out["boxes"].cpu()[keep].numpy().tolist()
        labels = out["labels"].cpu()[keep].numpy().tolist()
        scores = out["scores"].cpu()[keep].numpy().tolist()

        scale_x = orig_w / img_size
        scale_y = orig_h / img_size
        boxes_orig = [
            [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            for b in boxes
        ]

        total_detections += len(boxes_orig)

        # Draw and save
        annotated = draw_detections(img_bgr, boxes_orig, labels, scores)
        out_file = out_path / f"{img_p.stem}_detected.jpg"
        cv2.imwrite(str(out_file), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        dets = []
        for box, lbl, score in zip(boxes_orig, labels, scores):
            dets.append({
                "class_name": _cls_name(lbl),
                "class_id": lbl,
                "confidence": round(score, 4),
                "bbox": [round(v, 1) for v in box],
            })

        all_results.append({
            "image_name": img_p.name,
            "detections_count": len(dets),
            "inference_ms": round(dt_ms, 1),
            "detections": dets,
        })

        print(f"  [{idx:3d}/{len(image_files)}] {img_p.name:<30} -> {len(boxes_orig):3d} objects ({dt_ms:.1f} ms)")

    # Save summary JSON
    summary_json = out_path / "batch_predictions.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "total_images": len(all_results),
            "total_detections": total_detections,
            "confidence_threshold": confidence_thresh,
            "average_ms_per_image": round(total_time_ms / max(1, len(all_results)), 1),
            "fps": round(1000.0 / (total_time_ms / max(1, len(all_results))), 1) if total_time_ms > 0 else 0,
            "results": all_results,
        }, f, indent=2)

    avg_ms = total_time_ms / max(1, len(all_results))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"BATCH INFERENCE COMPLETED:")
    print(f"  Images processed: {len(all_results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average speed   : {avg_ms:.1f} ms/image ({fps:.1f} FPS)")
    print(f"  Saved JSON      : {summary_json}")
    print(f"  Saved Images    : {out_path.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multiple Image Directory UAV Object Detection Batch Tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", "--model", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--input_dir", default=None, help="Directory containing images")
    parser.add_argument("--output_dir", default=None, help="Output directory for annotated images")
    parser.add_argument("--confidence", "--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--experiment", default="final_model", choices=list(EXPERIMENT_MAP), help="Model architecture")
    parser.add_argument("--img_size", type=int, default=1280, help="Inference resolution")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    args = parser.parse_args()

    run_batch_folder_inference(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        confidence_thresh=args.confidence,
        experiment=args.experiment,
        img_size=args.img_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
