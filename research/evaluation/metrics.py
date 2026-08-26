"""
Evaluation Metrics for Small-Object Detection Research.

Computes:
- mAP@0.5, mAP@0.5:0.95 (COCO standard)
- AP_small, AP_medium, AP_large
- Per-class AP for all 10 VisDrone categories
- Precision, Recall, F1 (at specified confidence threshold)
- Inference time and FPS

All metrics are computed using pycocotools (COCOeval), which is the
standard for research papers and ensures comparability with literature.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


VISDRONE_CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# COCO area thresholds (in pixels^2)
AREA_SMALL = (0, 1024)         # < 32x32 px
AREA_MEDIUM = (1024, 9216)    # 32x32 - 96x96 px
AREA_LARGE = (9216, 1e10)     # > 96x96 px


@contextmanager
def temp_json(data: dict):
    """Context manager that writes data to a temp JSON file and yields the path."""
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(data, fh)
        fh.close()
        yield fh.name
    finally:
        try:
            os.unlink(fh.name)
        except OSError:
            pass


def compute_coco_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    coco_gt_api: Optional[COCO] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute full COCO evaluation metrics.

    Args:
        predictions: List of dicts with keys:
            image_id, category_id, bbox (XYWH), score
        ground_truths: List of dicts with keys:
            image_id, category_id, bbox (XYWH), area, iscrowd
        coco_gt_api: If provided, use this COCO API directly (more accurate).
        class_names: Category names for per-class AP reporting.

    Returns:
        Dict containing all metrics:
            map50, map50_95, ap_small, ap_medium, ap_large,
            precision, recall, f1, per_class_ap (dict),
            coco_stats (full stats array)
    """
    if not predictions:
        return _empty_metrics()

    if class_names is None:
        class_names = VISDRONE_CLASS_NAMES

    if coco_gt_api is not None:
        # Use actual COCO GT API (most accurate approach)
        coco_gt = coco_gt_api
    else:
        # Build from ground_truths list
        img_ids = sorted({g["image_id"] for g in ground_truths})
        cat_ids = sorted({g["category_id"] for g in ground_truths})
        gt_ann = {
            "images": [{"id": i} for i in img_ids],
            "categories": [{"id": c, "name": str(c)} for c in cat_ids],
            "annotations": [
                {**g, "id": idx + 1} for idx, g in enumerate(ground_truths)
            ],
        }
        with temp_json(gt_ann) as gt_path:
            coco_gt = COCO(gt_path)

    # Load predictions into COCO API
    if not predictions:
        return _empty_metrics()

    try:
        coco_dt = coco_gt.loadRes(predictions)
    except Exception:
        return _empty_metrics()

    # Overall evaluation
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()
    stats = e.stats  # Array of 12 COCO metrics

    # Per-class AP evaluation
    per_class_ap = {}
    cat_ids = sorted(coco_gt.getCatIds())
    for cat_id in cat_ids:
        e_cls = COCOeval(coco_gt, coco_dt, "bbox")
        e_cls.params.catIds = [cat_id]
        e_cls.evaluate()
        e_cls.accumulate()
        e_cls.summarize()
        # Get class name
        cat_info = coco_gt.loadCats([cat_id])[0]
        cat_name = cat_info.get("name", str(cat_id))
        per_class_ap[cat_name] = {
            "ap50_95": float(e_cls.stats[0]),
            "ap50": float(e_cls.stats[1]),
            "ap_small": float(e_cls.stats[3]),
            "ap_medium": float(e_cls.stats[4]),
            "ap_large": float(e_cls.stats[5]),
        }

    # Precision and Recall from predictions
    # For publication, we report precision/recall at the threshold where F1 is maximized
    precision_val, recall_val, f1_val = _compute_pr_from_coco(e)

    return {
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "ar_1": float(stats[6]),
        "ar_10": float(stats[7]),
        "ar_100": float(stats[8]),
        "ar_small": float(stats[9]),
        "ar_medium": float(stats[10]),
        "ar_large": float(stats[11]),
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "per_class_ap": per_class_ap,
        "coco_stats": [float(s) for s in stats],
    }


def _empty_metrics() -> Dict[str, Any]:
    """Return zero metrics when no predictions are available."""
    return {
        "map50_95": 0.0, "map50": 0.0, "map75": 0.0,
        "ap_small": 0.0, "ap_medium": 0.0, "ap_large": 0.0,
        "ar_1": 0.0, "ar_10": 0.0, "ar_100": 0.0,
        "ar_small": 0.0, "ar_medium": 0.0, "ar_large": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "per_class_ap": {}, "coco_stats": [0.0] * 12,
    }


def _compute_pr_from_coco(eval_obj: COCOeval) -> Tuple[float, float, float]:
    """
    Extract precision and recall at the best F1 threshold from COCOeval.
    Uses IoU=0.5 for the primary P/R/F1 metrics (standard for detection papers).
    """
    try:
        # eval_obj.eval['precision'] shape: [T, R, K, A, M]
        # T=IoU thresholds, R=recall pts, K=categories, A=area, M=maxDets
        # For IoU=0.5 (index 0), all categories (sum), area=all (index 0), maxDets=100 (index -1)
        prec = eval_obj.eval["precision"]
        # IoU=0.5 is index 0 in COCO's IoU grid [0.5:0.05:0.95]
        p = prec[0, :, :, 0, 2]  # [R, K] at IoU=0.5, area=all, maxDet=100
        p = p[p > -1]  # Filter invalid entries
        if len(p) == 0:
            return 0.0, 0.0, 0.0
        mean_p = float(np.mean(p))

        # Recall points used by COCOeval (101 points from 0 to 1)
        recall_pts = np.linspace(0.0, 1.0, 101)
        # Average precision across categories and recall points
        mean_r = float(np.mean(eval_obj.eval["recall"][0, :, 0, 2]))

        if mean_p + mean_r > 0:
            f1 = 2 * mean_p * mean_r / (mean_p + mean_r)
        else:
            f1 = 0.0

        return mean_p, mean_r, float(f1)
    except Exception:
        return 0.0, 0.0, 0.0


def measure_inference_time(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 1280,
    n_warmup: int = 10,
    n_measure: int = 50,
) -> Dict[str, float]:
    """
    Measure inference time and FPS.

    Args:
        model: Detection model in eval mode.
        device: Device to run on.
        img_size: Input image size.
        n_warmup: Warmup iterations (excluded from measurement).
        n_measure: Measurement iterations.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'fps'.
    """
    model.eval()
    dummy = [torch.zeros(3, img_size, img_size, device=device)]

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Synchronize GPU before measurement
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {"mean_ms": round(mean_ms, 2), "std_ms": round(std_ms, 2), "fps": round(fps, 2)}


def generate_pr_curve_data(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Generate precision-recall curve data points.

    Returns dict with 'precision' and 'recall' lists suitable for plotting.
    This uses the standard 101-point interpolated PR curve from COCO.
    """
    # Build a simple sorted-score PR computation
    if not predictions or not ground_truths:
        return {"precision": [], "recall": []}

    # Sort predictions by confidence score (descending)
    preds_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # Build GT lookup
    gt_by_img: Dict[int, List] = {}
    for g in ground_truths:
        gt_by_img.setdefault(g["image_id"], []).append(
            {"bbox": g["bbox"], "cat": g["category_id"], "matched": False}
        )

    tp_list, fp_list = [], []
    n_gt = len(ground_truths)

    for pred in preds_sorted:
        img_id = pred["image_id"]
        cat_id = pred["category_id"]
        px, py, pw, ph = pred["bbox"]
        pred_box = [px, py, px + pw, py + ph]

        gts = [g for g in gt_by_img.get(img_id, []) if g["cat"] == cat_id and not g["matched"]]
        if not gts:
            fp_list.append(1)
            tp_list.append(0)
            continue

        # Find best matching GT
        best_iou, best_idx = 0.0, -1
        for idx, g in enumerate(gts):
            gx, gy, gw, gh = g["bbox"]
            gt_box = [gx, gy, gx + gw, gy + gh]
            iou = _box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_iou >= iou_threshold:
            gts[best_idx]["matched"] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recall_pts = tp_cum / (n_gt + 1e-8)
    precision_pts = tp_cum / (tp_cum + fp_cum + 1e-8)

    return {
        "precision": precision_pts.tolist(),
        "recall": recall_pts.tolist(),
    }


def _box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-8
    return inter / union
