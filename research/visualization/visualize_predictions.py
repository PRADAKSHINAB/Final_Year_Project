"""
Qualitative Result Visualization for Research Paper.

Generates:
1. Prediction overlays: Input | GT | Prediction (side-by-side)
2. Difficult case analysis: tiny, crowded, overlapping, clutter
3. Model comparison grids: Baseline vs. Attention vs. SACM vs. Final
4. PR curves (matplotlib)
5. Loss + metric curves from training CSV
6. Per-class AP bar chart
7. Object size performance comparison chart

All plots are saved to results/visualizations/
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# VisDrone class colors (BGR for OpenCV)
CLASS_COLORS = {
    "pedestrian":      (0,   220, 255),
    "people":          (52,  211, 153),
    "bicycle":         (250, 204,  21),
    "car":             (244,  63,  94),
    "van":             (59,  130, 246),
    "truck":           (168,  85, 247),
    "tricycle":        (16,  185, 129),
    "awning-tricycle": (249, 115,  22),
    "bus":             (34,  197,  94),
    "motor":           (20,  184, 166),
}

VISDRONE_CLASSES = [
    "__background__",
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    color_by_class: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image (in-place copy)."""
    out = image.copy()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = VISDRONE_CLASSES[label] if 0 <= label < len(VISDRONE_CLASSES) else f"cls_{label}"
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        # Label with confidence
        score_str = f" {scores[i]:.2f}" if scores is not None else ""
        label_text = f"{cls_name}{score_str}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label_text, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


@torch.inference_mode()
def visualize_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    experiment_dir: Path,
    model_name: str,
    device: str = "cuda",
    num_images: int = 20,
    conf_thresh: float = 0.3,
) -> None:
    """
    Generate prediction visualization images for paper qualitative results.

    Saves side-by-side images: original | ground_truth | prediction
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    vis_dir = experiment_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for images, targets in val_loader:
        if saved >= num_images:
            break

        images_dev = [im.to(dev) for im in images]
        outputs = model(images_dev)

        for img_tensor, target, output in zip(images, targets, outputs):
            if saved >= num_images:
                break

            # Convert tensor to numpy image (HWC, uint8, RGB)
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Ground truth
            gt_img = draw_boxes(
                img_bgr,
                target["boxes"].numpy().tolist(),
                target["labels"].numpy().tolist(),
                scores=None,
                color_by_class=True,
            )

            # Predictions (filtered by confidence)
            keep = output["scores"].cpu() >= conf_thresh
            pred_boxes = output["boxes"].cpu()[keep].numpy().tolist()
            pred_labels = output["labels"].cpu()[keep].numpy().tolist()
            pred_scores = output["scores"].cpu()[keep].numpy().tolist()

            pred_img = draw_boxes(
                img_bgr,
                pred_boxes,
                pred_labels,
                scores=pred_scores,
                color_by_class=True,
            )

            # Add label headers
            h, w = img_bgr.shape[:2]
            header_h = 30
            for name, panel in [("INPUT", img_bgr), ("GROUND TRUTH", gt_img), ("PREDICTION", pred_img)]:
                header = np.zeros((header_h, w, 3), dtype=np.uint8)
                cv2.putText(header, name, (w // 2 - 60, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Stack header onto panel (not in-place)

            # Side-by-side with headers
            panels = []
            for name, panel in [("INPUT", img_bgr), ("GROUND TRUTH", gt_img), ("PREDICTION", pred_img)]:
                header = np.zeros((header_h, w, 3), dtype=np.uint8)
                cv2.putText(header, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(np.vstack([header, panel]))

            combined = np.hstack(panels)

            img_id = int(target["image_id"][0].item())
            out_path = vis_dir / f"{model_name}_img{img_id:06d}.jpg"
            cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

    print(f"[VIS] Saved {saved} prediction visualizations to {vis_dir}")


def plot_training_curves(
    csv_path: Path,
    output_dir: Path,
    model_name: str = "",
) -> None:
    """
    Generate training curve plots from CSV log.

    Generates:
    - loss_curve.png
    - map_curve.png
    - ap_size_curve.png (AP_small vs AP_medium vs AP_large)
    - lr_curve.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available; skipping curve plots.")
        return

    if not csv_path.exists():
        print(f"[WARNING] CSV not found: {csv_path}; skipping curve plots.")
        return

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v})

    if not rows:
        return

    epochs = [r["epoch"] for r in rows]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in [("loss_total", "Total"), ("loss_rpn_cls", "RPN Cls"),
                       ("loss_rpn_box", "RPN Box"), ("loss_roi_cls", "ROI Cls"),
                       ("loss_roi_box", "ROI Box")]:
        if key in rows[0]:
            ax.plot(epochs, [r[key] for r in rows], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    # mAP curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in [("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95")]:
        if key in rows[0]:
            ax.plot(epochs, [r[key] for r in rows], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title(f"mAP Curves — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_curve.png", dpi=150)
    plt.close()

    # AP by size curve
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in [("ap_small", "AP_small"), ("ap_medium", "AP_medium"), ("ap_large", "AP_large")]:
        if key in rows[0]:
            ax.plot(epochs, [r[key] for r in rows], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_title(f"AP by Object Size — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ap_size_curve.png", dpi=150)
    plt.close()

    print(f"[VIS] Training curves saved to {output_dir}")


def plot_ablation_comparison(
    ablation_csv: Path,
    output_dir: Path,
) -> None:
    """Generate ablation comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if not ablation_csv.exists():
        print("[WARNING] ABLATION_RESULTS.csv not found; skipping ablation plot.")
        return

    rows = []
    with open(ablation_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    models = [r.get("Model", f"Model {i}") for i, r in enumerate(rows)]
    metrics = ["mAP50", "mAP50-95", "AP_small"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(models))
    width = 0.5

    for ax, metric in zip(axes, metrics):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(metric, 0.0)))
            except (ValueError, TypeError):
                vals.append(0.0)
        bars = ax.bar(x, vals, width, color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(vals)])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", fontsize=8)

    plt.suptitle("Ablation Study Comparison", fontsize=13)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150)
    plt.close()
    print(f"[VIS] Ablation plot saved to {output_dir}")
