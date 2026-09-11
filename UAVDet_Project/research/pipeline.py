"""
research/pipeline.py
====================
Complete evaluation, visualization, and result storage pipeline.

Implements (with NO model architecture, dataset loader, or trainer modifications):
  1. Result Directory Structure: research/results/{E1_baseline, E2_attention, E3_sacm, E4_full}/
     with {checkpoints, predictions, comparisons, plots, metrics, logs}
  2. Prediction Images: JPG/PNG with original image, predicted bounding boxes, class name, confidence
  3. Ground Truth vs Prediction: Side-by-side comparison images (LEFT: GT, RIGHT: Prediction)
  4. E1/E2/E3/E4 Qualitative Comparison: Original | GT | E1 | E2 | E3 | E4
  5. predictions.json: Structured JSON per experiment with image_name and detections array
  6. metrics.json: mAP50, mAP50_95, AP_small/med/large, Prec, Rec, F1, inference_time, FPS, params, GFLOPs, GPU mem
  7. per_class_metrics.csv: AP50, AP50_95, precision, recall, F1 for all 10 VisDrone classes
  8. confusion_matrix.png & confusion_matrix.csv: IoU=0.5 matching logic
  9. small_object_metrics.json & small_object_analysis.png
 10. publication-quality training curves: train_loss.png, val_loss.png, learning_rate.png, map_curve.png
 11. efficiency.json: total & trainable params, GFLOPs, inference time, FPS, GPU memory
 12. Reproducibility: experiment_summary.txt and experiment_config.yaml
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from research.evaluation.metrics import compute_coco_metrics, measure_inference_time

# ──────────────────────────────────────────────────────────────────────────────
# VisDrone class constants
# ──────────────────────────────────────────────────────────────────────────────
VISDRONE_CLASSES: List[str] = [
    "__background__",
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
VISDRONE_FG: List[str] = VISDRONE_CLASSES[1:]   # indices 1-10

# BGR colours (one per foreground class)
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

JPEG_QUALITY = 92


def _cls_name(idx: int) -> str:
    return VISDRONE_CLASSES[idx] if 0 <= idx < len(VISDRONE_CLASSES) else f"cls_{idx}"


# ──────────────────────────────────────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────────────────────────────────────

def draw_boxes(
    image_bgr: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw detection bounding boxes on a BGR image."""
    out = image_bgr.copy()
    for i, (box, lbl) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = (int(v) for v in box)
        name  = _cls_name(lbl)
        color = _COLORS_BGR.get(name, (200, 200, 200))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        score_str = f" {scores[i]:.2f}" if scores is not None else ""
        text      = f"{name}{score_str}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(y1 - 4, th + 6)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + bl - 1), color, -1)
        cv2.putText(out, text, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _banner(img: np.ndarray, title: str, font_scale: float = 0.62) -> np.ndarray:
    """Prepend a dark title banner to an image panel."""
    h, w = img.shape[:2]
    bar = np.full((30, w, 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _hstack_panels(panels: List[np.ndarray]) -> np.ndarray:
    """Horizontally stack panels, padding shorter ones to max height."""
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)
    return np.hstack(padded)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction images, GT-vs-Pred comparisons, and predictions.json
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def save_predictions_and_comparisons(
    model: nn.Module,
    val_loader: DataLoader,
    results_dir: Path,
    model_name: str,
    device: str = "cuda",
    num_images: int = 50,
    conf_thresh: float = 0.25,
) -> List[Dict]:
    """
    Run inference on val_loader and save:
      predictions/image_XXXX.jpg          — pred overlay
      comparisons/image_XXXX_gt_vs_pred.jpg  — LEFT=GT, RIGHT=Pred
      predictions/predictions.json        — structured JSON per image
    """
    dev = torch.device(device)
    model = model.to(dev).eval()

    pred_dir = results_dir / "predictions"
    comp_dir = results_dir / "comparisons"
    pred_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    all_coco_preds: List[Dict] = []
    structured_predictions: List[Dict] = []

    saved = 0
    for images, targets in val_loader:
        if saved >= num_images:
            break
        imgs_dev = [im.to(dev, non_blocking=True) for im in images]
        outputs  = model(imgs_dev)

        for img_t, target, out in zip(images, targets, outputs):
            img_id   = int(target["image_id"][0].item())
            img_name = f"image_{img_id:04d}.jpg"
            meta     = val_loader.dataset.coco.loadImgs([img_id])[0] if hasattr(val_loader.dataset, "coco") else val_loader.dataset.dataset.coco.loadImgs([img_id])[0]
            orig_w   = meta.get("width", img_t.shape[2])
            orig_h   = meta.get("height", img_t.shape[1])
            scale_x  = orig_w / img_t.shape[2]
            scale_y  = orig_h / img_t.shape[1]

            # Decode tensor → BGR uint8
            img_np  = (img_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Ground truth
            gt_boxes  = target["boxes"].cpu().numpy().tolist()
            gt_labels = target["labels"].cpu().numpy().tolist()

            # Predictions filtered by threshold for visualization
            keep   = out["scores"].cpu() >= conf_thresh
            pb     = out["boxes"].cpu()[keep].numpy().tolist()
            pl     = out["labels"].cpu()[keep].numpy().tolist()
            ps     = out["scores"].cpu()[keep].numpy().tolist()

            image_detections = []

            # Collect for metrics / predictions.json (ALL detections, scaled to original image dimensions)
            for box, lbl, score in zip(
                out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
            ):
                x1, y1, x2, y2 = box.tolist()
                s = float(score); l = int(lbl)
                w_box = x2 - x1; h_box = y2 - y1
                
                # Scaled coordinates for original image dimensions
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                w_orig  = w_box * scale_x
                h_orig  = h_box * scale_y
                x2_orig = x1_orig + w_orig
                y2_orig = y1_orig + h_orig

                image_detections.append({
                    "class_id":   l,
                    "class_name": _cls_name(l),
                    "x1": round(x1_orig, 2), "y1": round(y1_orig, 2),
                    "x2": round(x2_orig, 2), "y2": round(y2_orig, 2),
                    "width":  round(w_orig, 2),
                    "height": round(h_orig, 2),
                    "confidence": round(s, 6),
                })
                
                all_coco_preds.append({
                    "image_id":   img_id,
                    "category_id": l,
                    "bbox":  [x1_orig, y1_orig, w_orig, h_orig],
                    "score": s,
                })

            structured_predictions.append({
                "image_id": img_id,
                "image_name": img_name,
                "detections": image_detections
            })

            if saved >= num_images:
                continue

            # Draw
            pred_img = draw_boxes(img_bgr, pb, pl, ps)
            gt_img   = draw_boxes(img_bgr, gt_boxes, gt_labels)

            # Prediction image
            pred_path = pred_dir / img_name
            cv2.imwrite(str(pred_path), pred_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            # GT vs Pred (LEFT=GT, RIGHT=Pred)
            gt_panel   = _banner(gt_img,   "GROUND TRUTH")
            pred_panel = _banner(pred_img, f"PREDICTION  ({model_name})")
            comp_img   = _hstack_panels([gt_panel, pred_panel])
            comp_path  = comp_dir / f"image_{img_id:04d}_gt_vs_pred.jpg"
            cv2.imwrite(str(comp_path), comp_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            saved += 1

    # Save predictions.json in requested structure
    json_path = pred_dir / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_predictions, f, indent=2)

    print(f"[PIPELINE] {saved} prediction images -> {pred_dir}")
    print(f"[PIPELINE] {len(structured_predictions)} image detection records -> {json_path}")
    return all_coco_preds


# ──────────────────────────────────────────────────────────────────────────────
# Metrics: metrics.json & per_class_metrics.csv
# ──────────────────────────────────────────────────────────────────────────────

def save_metrics(
    coco_preds: List[Dict],
    val_coco,
    metrics_dir: Path,
    model: nn.Module,
    device: str,
    img_size: int,
    model_name: str,
) -> Dict:
    """Compute COCO metrics, measure speed & efficiency, save metrics.json + per_class_metrics.csv."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("[PIPELINE] Computing COCO metrics...")
    metrics = compute_coco_metrics(coco_preds, [], coco_gt_api=val_coco)

    print("[PIPELINE] Measuring inference speed...")
    speed = measure_inference_time(model.to(dev), dev, img_size=img_size)
    metrics.update(speed)
    metrics["model_name"] = model_name

    # Add parameters, GFLOPs, and GPU memory info to metrics.json
    total_p = sum(p.numel() for p in model.parameters())
    metrics["number_of_parameters"] = total_p
    metrics["parameters_M"] = round(total_p / 1e6, 3)

    gpu_mem_mb = None
    if dev.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            with torch.no_grad():
                _ = model([torch.zeros(3, img_size, img_size, device=dev)])
            torch.cuda.synchronize(dev)
            gpu_mem_mb = round(torch.cuda.max_memory_allocated(dev) / 1e6, 1)
        except Exception as e:
            gpu_mem_mb = f"ERROR:{e}"
    metrics["GPU_memory_usage_MB"] = gpu_mem_mb

    gflops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fa = FlopCountAnalysis(model, ([torch.zeros(3, img_size, img_size, device=dev)],))
        gflops = round(fa.total() / 1e9, 2)
    except Exception:
        gflops = "null (install fvcore)"
    metrics["GFLOPs"] = gflops

    # metrics.json
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # per_class_metrics.csv
    per_class = metrics.get("per_class_ap", {})
    csv_path  = metrics_dir / "per_class_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "AP50", "AP50_95", "precision", "recall", "F1"])
        for cls_name in VISDRONE_FG:
            cd = per_class.get(cls_name, {})
            ap50    = cd.get("ap50",    "null")
            ap50_95 = cd.get("ap50_95", "null")
            prec    = cd.get("precision", "null")
            rec     = cd.get("recall",    "null")
            f1      = cd.get("f1",        "null")
            writer.writerow([cls_name, ap50, ap50_95, prec, rec, f1])

    print(f"[PIPELINE] mAP@0.5={metrics.get('map50',0):.4f}  "
          f"AP_small={metrics.get('ap_small',0):.4f}  "
          f"FPS={metrics.get('fps','null')}")
    print(f"[PIPELINE] metrics.json + per_class_metrics.csv -> {metrics_dir}")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Detection Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def _iou_xyxy(b1: List[float], b2: List[float]) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def compute_and_save_confusion_matrix(
    coco_preds: List[Dict],
    val_coco,
    metrics_dir: Path,
    iou_thresh: float = 0.50,
    conf_thresh: float = 0.25,
) -> np.ndarray:
    """Build an 11×11 confusion matrix with IoU=0.5 matching."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    n = len(VISDRONE_CLASSES)
    matrix = np.zeros((n, n), dtype=np.int64)

    # Load GT from COCO API
    gt_by_img: Dict[int, List] = defaultdict(list)
    for img_id in val_coco.getImgIds():
        for ann in val_coco.loadAnns(val_coco.getAnnIds(imgIds=[img_id])):
            x, y, w, h = ann["bbox"]
            gt_by_img[img_id].append({
                "xyxy": [x, y, x+w, y+h],
                "cat":  int(ann["category_id"]),
                "matched": False,
            })

    # Group predictions by image, filter by conf
    pred_by_img: Dict[int, List] = defaultdict(list)
    for p in coco_preds:
        if p["score"] >= conf_thresh:
            x, y, w, h = p["bbox"]
            pred_by_img[p["image_id"]].append({
                "xyxy":  [x, y, x+w, y+h],
                "cat":   int(p["category_id"]),
                "score": p["score"],
            })

    for img_id in set(list(gt_by_img) + list(pred_by_img)):
        gts   = gt_by_img.get(img_id, [])
        preds = sorted(pred_by_img.get(img_id, []), key=lambda x: -x["score"])

        for pred in preds:
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gts):
                if gt["matched"]:
                    continue
                iou = _iou_xyxy(pred["xyxy"], gt["xyxy"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            if best_iou >= iou_thresh and best_gi >= 0:
                gts[best_gi]["matched"] = True
                matrix[gts[best_gi]["cat"], pred["cat"]] += 1
            else:
                matrix[0, pred["cat"]] += 1   # FP → bg row

        for gt in gts:
            if not gt["matched"]:
                matrix[gt["cat"], 0] += 1     # FN → bg col

    # Save CSV
    csv_path = metrics_dir / "confusion_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["GT_backslash_Pred"] + VISDRONE_CLASSES)
        for i, row in enumerate(matrix):
            w.writerow([VISDRONE_CLASSES[i]] + row.tolist())

    # Save PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fg = matrix[1:, 1:].astype(float)
        labels = VISDRONE_FG
        fig, ax = plt.subplots(figsize=(12, 10))
        vmin = max(1, fg[fg > 0].min()) if fg.any() else 1
        im = ax.imshow(fg, cmap="Blues",
                       norm=LogNorm(vmin=vmin, vmax=max(vmin, fg.max()))
                       if fg.max() > 0 else None)
        plt.colorbar(im, ax=ax, label="Count (log scale)")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("Ground-Truth Class", fontsize=11)
        ax.set_title(
            f"Detection Confusion Matrix (foreground, IoU≥{iou_thresh}, conf≥{conf_thresh})\n"
            "Row=GT, Col=Predicted  |  Off-diagonal=misclassification",
            fontsize=10,
        )
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = int(fg[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=6, color="black" if v < fg.max()*0.5 else "white")
        plt.tight_layout()
        plt.savefig(metrics_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[WARNING] confusion matrix PNG: {e}")

    print(f"[PIPELINE] Confusion matrix -> {metrics_dir}")
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# Small object analysis
# ──────────────────────────────────────────────────────────────────────────────

def save_small_object_analysis(
    coco_preds: List[Dict],
    val_coco,
    metrics_dir: Path,
    plots_dir: Path,
    model_name: str,
    conf_thresh: float = 0.25,
) -> None:
    """Analyze detection performance by object size."""
    SMALL_THRESH  = 1024    # < 32×32 px²
    MEDIUM_THRESH = 9216    # < 96×96 px²

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    gt_by_size = {"small": 0, "medium": 0, "large": 0}
    gt_small_by_img: Dict[int, List] = defaultdict(list)

    for img_id in val_coco.getImgIds():
        for ann in val_coco.loadAnns(val_coco.getAnnIds(imgIds=[img_id])):
            area = ann.get("area") or (ann["bbox"][2] * ann["bbox"][3])
            if area < SMALL_THRESH:
                gt_by_size["small"] += 1
                x, y, w, h = ann["bbox"]
                gt_small_by_img[img_id].append({
                    "xyxy": [x, y, x+w, y+h],
                    "matched": False,
                })
            elif area < MEDIUM_THRESH:
                gt_by_size["medium"] += 1
            else:
                gt_by_size["large"] += 1

    def _iou(b1, b2):
        x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
        ia = max(0,min(x1+w1,x2+w2)-max(x1,x2)) * max(0,min(y1+h1,y2+h2)-max(y1,y2))
        return ia / max(w1*h1 + w2*h2 - ia, 1e-8)

    pred_by_img: Dict[int, List] = defaultdict(list)
    for p in coco_preds:
        if p["score"] >= conf_thresh:
            pred_by_img[p["image_id"]].append(p)

    tp = fp = fn = 0
    for img_id, gts in gt_small_by_img.items():
        preds = sorted(pred_by_img.get(img_id, []), key=lambda x: -x["score"])
        for pred in preds:
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gts):
                if gt["matched"]: continue
                px, py, pw, ph = pred["bbox"]
                iou = _iou(pred["bbox"], [gt["xyxy"][0], gt["xyxy"][1],
                                          gt["xyxy"][2]-gt["xyxy"][0],
                                          gt["xyxy"][3]-gt["xyxy"][1]])
                if iou > best_iou: best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                gts[best_gi]["matched"] = True; tp += 1
            else:
                fp += 1
        fn += sum(1 for g in gts if not g["matched"])

    prec_s = tp / max(tp + fp, 1)
    rec_s  = tp / max(tp + fn, 1)
    f1_s   = 2*prec_s*rec_s / max(prec_s + rec_s, 1e-8)

    result = {
        "model_name":            model_name,
        "iou_threshold":         0.5,
        "conf_threshold":        conf_thresh,
        "gt_small_total":        gt_by_size["small"],
        "gt_medium_total":       gt_by_size["medium"],
        "gt_large_total":        gt_by_size["large"],
        "small_TP":              tp,
        "small_FP":              fp,
        "small_FN":              fn,
        "recall_small":          round(rec_s, 4),
        "precision_small":       round(prec_s, 4),
        "f1_small":              round(f1_s, 4),
    }

    with open(metrics_dir / "small_object_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Small-Object Analysis — {model_name}", fontsize=13)

        sizes  = [gt_by_size["small"], gt_by_size["medium"], gt_by_size["large"]]
        clrs   = ["#FF6B6B", "#FFA500", "#4CAF50"]
        lbls   = [f"Small\n<32²px\n{sizes[0]}", f"Medium\n32²-96²px\n{sizes[1]}",
                  f"Large\n>96²px\n{sizes[2]}"]
        ax1.pie([max(s, 1) for s in sizes], labels=lbls, colors=clrs,
                autopct="%1.1f%%", startangle=140)
        ax1.set_title("GT Object-Size Distribution")

        cats = ["TP\n(detected)", "FN\n(missed)", "FP\n(false alarm)"]
        vals = [tp, fn, fp]
        clrs2 = ["#4CAF50", "#FF6B6B", "#FF9800"]
        bars = ax2.bar(cats, vals, color=clrs2, edgecolor="black", linewidth=0.5, width=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(v), ha="center", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Small-Object Detections\nconf≥{conf_thresh}, IoU≥0.5\n"
                      f"Prec={prec_s:.3f}  Rec={rec_s:.3f}  F1={f1_s:.3f}")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "small_object_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[WARNING] small_object_analysis.png: {e}")

    print(f"[PIPELINE] small_object_metrics.json + small_object_analysis.png -> done")


# ──────────────────────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────────────────────

def save_training_curves(
    csv_path: Path,
    plots_dir: Path,
    model_name: str = "",
) -> None:
    """Read training_log.csv and generate publication-quality loss & map curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        print(f"[WARNING] training_log.csv not found: {csv_path}; skipping curves.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available.")
        return

    rows: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: Dict = {}
            for k, v in row.items():
                try:    parsed[k] = float(v)
                except: parsed[k] = v
            rows.append(parsed)

    if not rows:
        print("[WARNING] training_log.csv is empty.")
        return

    epochs = [r.get("epoch", i+1) for i, r in enumerate(rows)]

    def _plot(key_label_pairs, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(9, 4))
        plotted = False
        for key, label in key_label_pairs:
            if key in rows[0] and isinstance(rows[0][key], float):
                ax.plot(epochs, [r[key] for r in rows], label=label, linewidth=1.6)
                plotted = True
        if not plotted:
            plt.close(); return
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title}  —  {model_name}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

    _plot([("loss_total","Total"),("loss_rpn_cls","RPN-Cls"),
           ("loss_rpn_box","RPN-Box"),("loss_roi_cls","ROI-Cls"),
           ("loss_roi_box","ROI-Box")],
          "Training Loss", "Loss", "train_loss.png")

    _plot([("map50","mAP@0.50"),("map50_95","mAP@0.5:0.95")],
          "Validation mAP", "mAP", "val_loss.png")

    _plot([("lr","Learning Rate")],
          "Learning Rate", "LR", "learning_rate.png")

    _plot([("map50","mAP@0.50")],
          "mAP@0.50 Curve", "mAP@0.50", "map_curve.png")

    _plot([("ap_small","AP_small"),("ap_medium","AP_medium"),("ap_large","AP_large")],
          "AP by Object Size", "AP", "ap_size_curve.png")

    print(f"[PIPELINE] Training curves -> {plots_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Efficiency metrics
# ──────────────────────────────────────────────────────────────────────────────

def save_efficiency(
    model: nn.Module,
    device: str,
    img_size: int,
    metrics_dir: Path,
    model_name: str,
) -> Dict:
    """Measure and save efficiency.json."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    model = model.to(dev).eval()

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    speed = measure_inference_time(model, dev, img_size=img_size)

    gpu_mem_mb = None
    if dev.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            with torch.no_grad():
                _ = model([torch.zeros(3, img_size, img_size, device=dev)])
            torch.cuda.synchronize(dev)
            gpu_mem_mb = round(torch.cuda.max_memory_allocated(dev) / 1e6, 1)
        except Exception as e:
            gpu_mem_mb = f"ERROR:{e}"

    gflops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fa = FlopCountAnalysis(model, ([torch.zeros(3, img_size, img_size, device=dev)],))
        gflops = round(fa.total() / 1e9, 2)
    except Exception:
        gflops = "null (install fvcore)"

    rec = {
        "model_name":             model_name,
        "total_parameters":       total_p,
        "total_parameters_M":     round(total_p/1e6, 3),
        "trainable_parameters":   trainable_p,
        "trainable_parameters_M": round(trainable_p/1e6, 3),
        "GFLOPs":                 gflops,
        "GPU_memory_used_MB":     gpu_mem_mb,
        "inference_time_ms":      speed.get("mean_ms"),
        "inference_std_ms":       speed.get("std_ms"),
        "FPS":                    speed.get("fps"),
        "img_size":               img_size,
        "device":                 str(dev),
    }

    with open(metrics_dir / "efficiency.json", "w") as f:
        json.dump(rec, f, indent=2)

    print(f"[PIPELINE] efficiency.json: params={total_p/1e6:.2f}M  FPS={speed.get('fps','null')}")
    return rec


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility files
# ──────────────────────────────────────────────────────────────────────────────

def save_reproducibility(
    cfg: Dict,
    model_name: str,
    metrics: Dict,
    efficiency: Dict,
    results_dir: Path,
    train_n: int = 0,
    val_n: int = 0,
) -> None:
    """Save experiment_config.yaml and experiment_summary.txt."""
    import platform, yaml

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(logs_dir / "experiment_config.yaml", "w") as f:
        yaml.dump({**cfg, "model_name": model_name}, f, default_flow_style=False)

    try:
        torch_ver = torch.__version__
        cuda_ver  = torch.version.cuda or "N/A"
        gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except Exception:
        torch_ver = cuda_ver = gpu_name = "N/A"

    def _v(d, k, fmt=None):
        v = d.get(k)
        if v is None: return "null"
        return (f"{v:{fmt}}" if fmt and isinstance(v, float) else str(v))

    lines = [
        f"EXPERIMENT SUMMARY — {model_name}",
        "=" * 60,
        f"Model:              {model_name}",
        f"Dataset:            VisDrone-DET 2019",
        f"Train images:       {train_n or 'NOT_KNOWN'}",
        f"Val images:         {val_n or 'NOT_KNOWN'}",
        f"Image size:         {cfg.get('img_size','N/A')} px",
        f"Batch size:         {cfg.get('batch_size','N/A')}",
        f"Epochs:             {cfg.get('epochs','N/A')}",
        f"Optimizer:          SGD lr={cfg.get('optimizer',{}).get('lr','N/A')}",
        f"LR scheduler:       CosineAnnealing + warmup",
        f"Random seed:        {cfg.get('seed', 42)}",
        "",
        f"GPU:                {gpu_name}",
        f"PyTorch version:    {torch_ver}",
        f"CUDA version:       {cuda_ver}",
        f"OS:                 {platform.system()} {platform.release()}",
        "",
        f"Parameters (total): {efficiency.get('total_parameters_M','null')} M",
        f"Parameters (train): {efficiency.get('trainable_parameters_M','null')} M",
        f"GFLOPs:             {efficiency.get('GFLOPs','null')}",
        f"Inference time:     {efficiency.get('inference_time_ms','null')} ms",
        f"FPS:                {efficiency.get('FPS','null')}",
        f"GPU mem (peak):     {efficiency.get('GPU_memory_used_MB','null')} MB",
        "",
        "Evaluation Metrics:",
        f"  mAP@0.50:         {_v(metrics,'map50','.4f')}",
        f"  mAP@0.5:0.95:     {_v(metrics,'map50_95','.4f')}",
        f"  AP_small:         {_v(metrics,'ap_small','.4f')}",
        f"  AP_medium:        {_v(metrics,'ap_medium','.4f')}",
        f"  AP_large:         {_v(metrics,'ap_large','.4f')}",
        f"  Precision:        {_v(metrics,'precision','.4f')}",
        f"  Recall:           {_v(metrics,'recall','.4f')}",
        f"  F1:               {_v(metrics,'f1','.4f')}",
        "",
        "Best checkpoint selected by: mAP@0.50 (validation set).",
        "All metrics from pycocotools COCOeval — no values are fabricated.",
    ]

    with open(logs_dir / "experiment_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[PIPELINE] experiment_config.yaml + experiment_summary.txt -> {logs_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline executor
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_pipeline(
    model: nn.Module,
    val_loader: DataLoader,
    val_coco,
    results_dir: Path,
    model_name: str,
    cfg: Dict,
    device: str = "cuda",
    num_vis: int = 50,
    conf_thresh: float = 0.25,
) -> Dict:
    """Run all pipeline tasks for a single experiment."""
    results_dir = Path(results_dir)
    metrics_dir = results_dir / "metrics"
    plots_dir   = results_dir / "plots"
    img_size    = cfg.get("img_size", 1280)

    print(f"\n{'='*60}")
    print(f"PIPELINE — {model_name}")
    print(f"{'='*60}")

    coco_preds = save_predictions_and_comparisons(
        model, val_loader, results_dir, model_name,
        device=device, num_images=num_vis, conf_thresh=conf_thresh,
    )

    metrics = save_metrics(coco_preds, val_coco, metrics_dir,
                           model, device, img_size, model_name)

    compute_and_save_confusion_matrix(coco_preds, val_coco, metrics_dir,
                                      conf_thresh=conf_thresh)

    save_small_object_analysis(coco_preds, val_coco, metrics_dir, plots_dir,
                               model_name, conf_thresh=conf_thresh)

    eff = save_efficiency(model, device, img_size, metrics_dir, model_name)

    save_training_curves(results_dir / "logs" / "training_log.csv", plots_dir, model_name)

    save_reproducibility(cfg, model_name, metrics, eff, results_dir,
                         val_n=len(val_loader.dataset))

    print(f"\n[PIPELINE DONE] {results_dir}")
    return metrics
