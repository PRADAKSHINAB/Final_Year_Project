"""
Full Evaluation Pipeline for Research Experiments.

Generates:
- Complete COCO metrics (mAP@0.5, mAP@0.5:0.95, AP_small/medium/large)
- Per-class AP table
- PR curves (saved as JSON for plotting)
- Loss curves (from CSV)
- Object size performance chart data
- Inference time and FPS
- metrics.json summary
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from research.evaluation.metrics import (
    compute_coco_metrics,
    generate_pr_curve_data,
    measure_inference_time,
)


def run_full_evaluation(
    model: nn.Module,
    val_loader: DataLoader,
    val_coco: Any,
    experiment_dir: Path,
    model_name: str,
    device: str = "cuda",
    img_size: int = 1280,
) -> Dict[str, Any]:
    """
    Run full evaluation and save all results.

    Args:
        model: Trained model.
        val_loader: Validation DataLoader.
        val_coco: COCO API for ground truth.
        experiment_dir: Directory to save results.
        model_name: Name of the model (for logging).
        device: Device string.
        img_size: Input image size (for FPS measurement).

    Returns:
        Full metrics dictionary.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    print(f"\n[EVAL] Running full evaluation for: {model_name}")
    print(f"       Validation samples: {len(val_loader.dataset)}")

    # Collect predictions
    predictions: List[Dict] = []
    ground_truths: List[Dict] = []

    t_start = time.time()
    with torch.inference_mode():
        for images, targets in val_loader:
            images = [im.to(dev, non_blocking=True) for im in images]
            outputs = model(images)

            for target, out in zip(targets, outputs):
                img_id = int(target["image_id"][0].item())

                # ---------------------------------------------------------
                # Convert model-output coordinates back to original image
                # coordinates before COCO evaluation.
                #
                # The model operates on img_size x img_size images, while
                # VisDrone COCO annotations use the original image
                # dimensions (e.g. 960x540, 1360x765).
                # ---------------------------------------------------------
                meta = val_coco.loadImgs([img_id])[0]
                orig_w = meta.get("width", img_size)
                orig_h = meta.get("height", img_size)

                scale_x = orig_w / img_size
                scale_y = orig_h / img_size

                # Predictions
                for box, label, score in zip(
                    out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
                ):
                    x1, y1, x2, y2 = box.tolist()

                    # Scale prediction from model coordinates to
                    # original VisDrone image coordinates.
                    x1_orig = x1 * scale_x
                    y1_orig = y1 * scale_y
                    x2_orig = x2 * scale_x
                    y2_orig = y2 * scale_y

                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(label.item()),
                        "bbox": [
                            x1_orig,
                            y1_orig,
                            x2_orig - x1_orig,
                            y2_orig - y1_orig,
                        ],
                        "score": float(score.item()),
                    })

                # Ground truths
                for box, label, area, iscrowd in zip(
                    target["boxes"], target["labels"],
                    target["area"], target["iscrowd"]
                ):
                    x1, y1, x2, y2 = box.tolist()
                    ground_truths.append({
                        "image_id": img_id,
                        "category_id": int(label.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float(area.item()),
                        "iscrowd": int(iscrowd.item()),
                    })

    eval_time = time.time() - t_start
    print(f"[EVAL] Collected {len(predictions)} predictions from {len(val_loader)} batches in {eval_time:.1f}s")

    # Compute COCO metrics
    print("[EVAL] Computing COCO metrics...")
    metrics = compute_coco_metrics(predictions, ground_truths, coco_gt_api=val_coco)

    # Measure inference speed
    print("[EVAL] Measuring inference speed...")
    speed = measure_inference_time(model, dev, img_size=img_size)
    metrics.update(speed)
    metrics["model_name"] = model_name

    # Generate PR curve data
    print("[EVAL] Generating PR curve data...")
    pr_data = generate_pr_curve_data(predictions, ground_truths, iou_threshold=0.5)

    # Save results
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # PR curve data
    with open(results_dir / "pr_curve_data.json", "w") as f:
        json.dump(pr_data, f)

    # Per-class AP as pretty table
    with open(results_dir / "per_class_ap.json", "w") as f:
        json.dump(metrics.get("per_class_ap", {}), f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS — {model_name}")
    print("=" * 60)
    print(f"  mAP@0.5        : {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics['map50_95']:.4f}")
    print(f"  AP_small       : {metrics['ap_small']:.4f}  *** PRIMARY METRIC ***")
    print(f"  AP_medium      : {metrics['ap_medium']:.4f}")
    print(f"  AP_large       : {metrics['ap_large']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1             : {metrics['f1']:.4f}")
    print(f"  Inference      : {metrics.get('mean_ms', 'N/A')} ms")
    print(f"  FPS            : {metrics.get('fps', 'N/A')}")
    print("=" * 60)

    if metrics.get("per_class_ap"):
        print("\nPer-Class AP@0.5:")
        for cls_name, cls_metrics in metrics["per_class_ap"].items():
            print(f"  {cls_name:20s}: {cls_metrics.get('ap50', 0):.4f}")

    print(f"\n[EVAL] Results saved to: {experiment_dir}")
    return metrics
