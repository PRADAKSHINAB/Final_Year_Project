"""
Error Analysis for Small-Object Detection Research.

Systematically identifies and categorizes detection failures:
1. Missed small objects (false negatives, area < 1024 px^2)
2. False positives (predictions with no matching GT)
3. Duplicate detections (multiple predictions for same GT)
4. Localization errors (IoU > 0.1 but < threshold)
5. Crowded-scene failures (objects with nearby neighbors)
6. Low-confidence predictions (correct class, low score)

Results are saved to ERROR_ANALYSIS.md and error_cases/ directory.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


SMALL_AREA_THRESH = 1024    # < 32x32 px
CROWD_DIST_THRESH = 50      # pixels — objects closer than this are "crowded"
IOU_MATCH_THRESH = 0.5
IOU_LOC_THRESH = 0.1        # IoU > 0.1 but < 0.5 = localization error


def _box_iou(b1: List[float], b2: List[float]) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-8
    return inter / union


def _xywh_to_xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _box_center(bbox_xyxy: List[float]) -> Tuple[float, float]:
    return (bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2


def analyze_errors(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresh: float = IOU_MATCH_THRESH,
) -> Dict[str, Any]:
    """
    Perform error analysis on detection results.

    Args:
        predictions: List of prediction dicts (image_id, category_id, bbox XYWH, score)
        ground_truths: List of GT dicts (image_id, category_id, bbox XYWH, area, iscrowd)
        iou_thresh: IoU threshold for matching (default 0.5)

    Returns:
        Error analysis summary dict.
    """
    # Organize by image
    gt_by_img: Dict[int, List] = defaultdict(list)
    pred_by_img: Dict[int, List] = defaultdict(list)

    for g in ground_truths:
        gt_by_img[g["image_id"]].append(g)
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)

    errors = {
        "missed_small": [],       # FN where area < SMALL_AREA_THRESH
        "missed_medium": [],      # FN where area in medium range
        "missed_large": [],       # FN where area > medium
        "false_positives": [],    # Predictions with IoU < 0.1 to any GT
        "localization_errors": [], # 0.1 < IoU < 0.5
        "duplicate_detections": [], # Multiple preds matching same GT
        "crowded_failures": [],   # Missed objects in dense regions
        "low_confidence_tp": [],  # True positives with score < 0.5
    }

    total_gt = 0
    total_pred = 0
    total_tp = 0

    for img_id in set(list(gt_by_img.keys()) + list(pred_by_img.keys())):
        gts = gt_by_img.get(img_id, [])
        preds = sorted(pred_by_img.get(img_id, []), key=lambda x: -x["score"])

        gt_boxes = [_xywh_to_xyxy(g["bbox"]) for g in gts]
        pred_boxes = [_xywh_to_xyxy(p["bbox"]) for p in preds]

        total_gt += len(gts)
        total_pred += len(preds)

        gt_matched = [False] * len(gts)
        gt_match_count = [0] * len(gts)

        for pi, (pred, pbox) in enumerate(zip(preds, pred_boxes)):
            best_iou = 0.0
            best_gi = -1
            for gi, gbox in enumerate(gt_boxes):
                iou = _box_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_thresh and best_gi >= 0:
                total_tp += 1
                gt_match_count[best_gi] += 1
                if not gt_matched[best_gi]:
                    gt_matched[best_gi] = True
                    if pred["score"] < 0.5:
                        errors["low_confidence_tp"].append({
                            "image_id": img_id, "score": pred["score"],
                            "category_id": pred.get("category_id")
                        })
                else:
                    # Already matched — duplicate detection
                    errors["duplicate_detections"].append({
                        "image_id": img_id, "score": pred["score"],
                        "gt_idx": best_gi, "iou": best_iou
                    })
            elif best_iou >= IOU_LOC_THRESH:
                errors["localization_errors"].append({
                    "image_id": img_id, "iou": best_iou, "score": pred["score"]
                })
            else:
                errors["false_positives"].append({
                    "image_id": img_id, "score": pred["score"],
                    "category_id": pred.get("category_id")
                })

        # Analyze missed GT objects
        for gi, (gt, gbox, matched) in enumerate(zip(gts, gt_boxes, gt_matched)):
            if not matched:
                area = gt.get("area", (gbox[2] - gbox[0]) * (gbox[3] - gbox[1]))
                cx, cy = _box_center(gbox)

                # Check if in crowded region
                nearby = sum(
                    1 for other_gt, other_box in zip(gts, gt_boxes)
                    if other_gt is not gt and
                    abs(cx - _box_center(other_box)[0]) < CROWD_DIST_THRESH and
                    abs(cy - _box_center(other_box)[1]) < CROWD_DIST_THRESH
                )
                is_crowded = nearby >= 2

                miss_info = {
                    "image_id": img_id, "area": area, "category_id": gt.get("category_id"),
                    "is_crowded": is_crowded,
                }

                if is_crowded:
                    errors["crowded_failures"].append(miss_info)
                elif area < SMALL_AREA_THRESH:
                    errors["missed_small"].append(miss_info)
                elif area < 9216:
                    errors["missed_medium"].append(miss_info)
                else:
                    errors["missed_large"].append(miss_info)

    summary = {
        "total_gt": total_gt,
        "total_predictions": total_pred,
        "total_tp": total_tp,
        "missed_small_count": len(errors["missed_small"]),
        "missed_medium_count": len(errors["missed_medium"]),
        "missed_large_count": len(errors["missed_large"]),
        "false_positive_count": len(errors["false_positives"]),
        "localization_error_count": len(errors["localization_errors"]),
        "duplicate_count": len(errors["duplicate_detections"]),
        "crowded_failure_count": len(errors["crowded_failures"]),
        "low_confidence_tp_count": len(errors["low_confidence_tp"]),
        "recall_small": 1.0 - (len(errors["missed_small"]) / max(1, sum(
            1 for g in ground_truths if g.get("area", 0) < SMALL_AREA_THRESH
        ))),
    }

    return {"summary": summary, "errors": {k: errors[k][:20] for k in errors}}  # Cap sample size


def save_error_analysis_report(
    analysis_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save ERROR_ANALYSIS.md comparing multiple models.

    Args:
        analysis_results: Dict mapping model_name -> analysis dict from analyze_errors()
        output_path: Path to write the markdown report.
    """
    lines = [
        "# Error Analysis Report",
        "",
        "## Overview",
        "",
        "This document analyzes failure cases across all experimental configurations.",
        "The primary focus is on missed small objects (area < 1024 px^2 = 32x32 pixels),",
        "as AP_small improvement is the main research claim.",
        "",
        "## Error Categories",
        "",
        "| Category | Definition |",
        "|----------|-----------|",
        "| Missed small | GT box with area < 1024 px^2 not detected |",
        "| False positive | Prediction with IoU < 0.1 to any GT |",
        "| Localization error | IoU between 0.1 and 0.5 (detected but inaccurate) |",
        "| Duplicate | Multiple predictions for same GT object |",
        "| Crowded failure | Missed object in dense region (≥2 neighbors within 50px) |",
        "| Low-confidence TP | Correct prediction with score < 0.5 |",
        "",
        "## Results by Model",
        "",
    ]

    if analysis_results:
        # Table header
        lines.append("| Metric | " + " | ".join(analysis_results.keys()) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(analysis_results)) + "|")

        metrics_to_show = [
            ("total_gt", "Total GT objects"),
            ("missed_small_count", "Missed small objects"),
            ("missed_medium_count", "Missed medium objects"),
            ("false_positive_count", "False positives"),
            ("localization_error_count", "Localization errors"),
            ("crowded_failure_count", "Crowded failures"),
            ("duplicate_count", "Duplicate detections"),
        ]
        for key, label in metrics_to_show:
            row = f"| {label} |"
            for model_name, result in analysis_results.items():
                val = result.get("summary", {}).get(key, "N/A")
                row += f" {val} |"
            lines.append(row)
    else:
        lines.append("*Error analysis results not yet available. Run experiments first.*")
        lines.append("")
        lines.append("## Expected findings after training:")
        lines.append("")
        lines.append("1. **Missed small objects**: Should decrease with attention module")
        lines.append("2. **False positives**: Should decrease with scale-aware RPN")
        lines.append("3. **Crowded failures**: Partially addressed by SACM context aggregation")
        lines.append("4. **Localization errors**: May improve with CA's spatial precision")

    lines += [
        "",
        "## Analysis Methodology",
        "",
        "All errors are computed using IoU threshold = 0.5 (COCO standard).",
        "Small object threshold = 1024 px^2 (COCO standard = 32x32 px).",
        "Crowded region defined as: ≥2 other objects within 50-pixel radius of center.",
        "",
        "## Limitations",
        "",
        "- Error counts depend on confidence threshold (default: all predictions, no threshold filter)",
        "- Crowded failure definition uses a fixed distance threshold",
        "- Category-specific errors require larger sample analysis",
        "",
        "## [RESULT TO BE FILLED AFTER TRAINING]",
        "",
        "The following comparisons will be added after all experiments are complete:",
        "",
        "- Baseline vs. Attention: Change in missed_small_count",
        "- Baseline vs. SACM: Change in false_positive_count",
        "- Full model vs. Baseline: Overall improvement breakdown",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Error analysis report saved to: {output_path}")
