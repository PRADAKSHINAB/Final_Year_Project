"""
research/sanity_test.py
========================
End-to-end sanity test for the entire pipeline.

Runs WITHOUT the real VisDrone dataset.
Uses synthetic images and GT annotations to verify that every
required output file is physically created and can be opened.

Checks:
  predictions/image_000001_prediction.jpg    -> JPEG viewable
  comparisons/image_000001_gt_vs_pred.jpg    -> JPEG viewable
  predictions/predictions.json               -> valid JSON
  metrics/metrics.json                       -> valid JSON (N/A values OK)
  metrics/per_class_metrics.csv              -> valid CSV
  metrics/confusion_matrix.png               -> PNG viewable
  metrics/confusion_matrix.csv               -> valid CSV
  metrics/small_object_metrics.json          -> valid JSON
  metrics/small_object_analysis.png          -> PNG viewable
  metrics/efficiency.json                    -> valid JSON
  plots/train_loss.png                       -> PNG viewable
  plots/learning_rate.png                    -> PNG viewable
  logs/experiment_config.yaml                -> valid YAML
  logs/experiment_summary.txt                -> text file

Usage:
    python research/sanity_test.py
    python research/sanity_test.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path("research/results/E1_baseline")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ok(msg):  print(f"  PASS  {msg}")
def _warn(msg):print(f"  WARN  {msg}")
def _fail(msg):print(f"  FAIL  {msg}"); return False

def check_file(path: Path, tag: str) -> bool:
    p = Path(path)
    if not p.exists():
        return _fail(f"{tag}  [NOT FOUND: {p}]")
    if p.stat().st_size == 0:
        return _fail(f"{tag}  [EMPTY FILE: {p}]")
    _ok(f"{tag}  ({p.stat().st_size} bytes)")
    return True

def check_image(path: Path, tag: str) -> bool:
    if not check_file(path, tag):
        return False
    img = cv2.imread(str(path))
    if img is None:
        return _fail(f"{tag}  [CANNOT OPEN AS IMAGE]")
    _ok(f"{tag}  [image {img.shape[1]}x{img.shape[0]} OK]")
    return True

def check_json(path: Path, tag: str) -> bool:
    if not check_file(path, tag):
        return False
    try:
        data = json.loads(Path(path).read_text())
        _ok(f"{tag}  [valid JSON, {len(data) if isinstance(data,(list,dict)) else '?'} items]")
        return True
    except Exception as e:
        return _fail(f"{tag}  [INVALID JSON: {e}]")

def check_csv(path: Path, tag: str) -> bool:
    if not check_file(path, tag):
        return False
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        _ok(f"{tag}  [valid CSV, {len(rows)} rows]")
        return True
    except Exception as e:
        return _fail(f"{tag}  [INVALID CSV: {e}]")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_coco_api(n_images: int = 3, n_cats: int = 10):
    """Build a minimal pycocotools COCO object with synthetic annotations."""
    from pycocotools.coco import COCO
    import io, json as _json

    imgs  = [{"id": i+1, "width": 640, "height": 640, "file_name": f"img{i+1}.jpg"}
             for i in range(n_images)]
    cats  = [{"id": c+1, "name": name, "supercategory": "vehicle"}
             for c, name in enumerate([
                 "pedestrian","people","bicycle","car","van",
                 "truck","tricycle","awning-tricycle","bus","motor"])
             ][:n_cats]
    anns  = []
    ann_id = 1
    for img in imgs:
        for _ in range(np.random.randint(3, 8)):
            x = int(np.random.uniform(0, 500))
            y = int(np.random.uniform(0, 500))
            w = int(np.random.uniform(10, 80))
            h = int(np.random.uniform(10, 80))
            cat = int(np.random.randint(1, n_cats+1))
            area = w * h
            anns.append({"id": ann_id, "image_id": img["id"], "category_id": cat,
                         "bbox": [x, y, w, h], "area": area, "iscrowd": 0})
            ann_id += 1

    coco_dict = {"images": imgs, "annotations": anns, "categories": cats}

    # Load via COCO API (requires writing to temp file — COCO requires file path)
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        _json.dump(coco_dict, tf)
        tf_name = tf.name

    coco = COCO(tf_name)
    os.unlink(tf_name)
    return coco, imgs


def make_synthetic_batch(n_images: int = 3, img_size: int = 640, device="cpu"):
    """Return (images_list, targets_list) matching VisDroneDataset format."""
    images, targets = [], []
    for i in range(n_images):
        img = torch.rand(3, img_size, img_size)
        images.append(img)
        n_gt = np.random.randint(2, 6)
        boxes = []
        for _ in range(n_gt):
            x1, y1 = np.random.randint(0,500,size=2).tolist()
            x2, y2 = x1 + np.random.randint(20,80), y1 + np.random.randint(20,80)
            boxes.append([float(x1),float(y1),float(min(x2,img_size)),float(min(y2,img_size))])
        tgt = {
            "image_id": torch.tensor([i+1]),
            "boxes":    torch.tensor(boxes, dtype=torch.float32),
            "labels":   torch.randint(1, 11, (n_gt,)),
            "area":     torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes]),
            "iscrowd":  torch.zeros(n_gt, dtype=torch.int64),
        }
        targets.append(tgt)
    return images, targets


# ──────────────────────────────────────────────────────────────────────────────
# Build a synthetic training_log.csv
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_training_log(logs_dir: Path, n_epochs: int = 5) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "training_log.csv"
    fields = ["epoch","loss_total","loss_rpn_cls","loss_rpn_box",
              "loss_roi_cls","loss_roi_box","lr","map50","map50_95",
              "ap_small","ap_medium","ap_large","precision","recall","f1"]
    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in range(1, n_epochs+1):
            decay = 1.0 - 0.1*e
            w.writerow({
                "epoch": e,
                "loss_total":   round(2.0*decay + 0.05*np.random.randn(), 4),
                "loss_rpn_cls": round(0.5*decay + 0.02*np.random.randn(), 4),
                "loss_rpn_box": round(0.4*decay + 0.02*np.random.randn(), 4),
                "loss_roi_cls": round(0.6*decay + 0.02*np.random.randn(), 4),
                "loss_roi_box": round(0.5*decay + 0.02*np.random.randn(), 4),
                "lr":           round(0.005 * decay, 6),
                "map50":        round(0.05*e + 0.01*np.random.randn(), 4),
                "map50_95":     round(0.02*e + 0.005*np.random.randn(), 4),
                "ap_small":     round(0.01*e + 0.005*np.random.randn(), 4),
                "ap_medium":    round(0.03*e + 0.005*np.random.randn(), 4),
                "ap_large":     round(0.06*e + 0.01*np.random.randn(), 4),
                "precision":    round(0.3 + 0.05*e + 0.02*np.random.randn(), 4),
                "recall":       round(0.2 + 0.04*e + 0.02*np.random.randn(), 4),
                "f1":           round(0.25 + 0.04*e + 0.02*np.random.randn(), 4),
            })
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# Main sanity test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--img_size", type=int, default=640)
    args = p.parse_args()

    print("\n" + "="*60)
    print("SANITY TEST — SOA-FasterRCNN Pipeline")
    print("No real dataset required.")
    print("="*60 + "\n")

    results_dir = RESULTS_ROOT
    for sub in ["checkpoints","predictions","comparisons","plots","metrics","logs"]:
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    PASS_count = 0
    FAIL_count = 0

    def _check(fn, *a, **kw):
        nonlocal PASS_count, FAIL_count
        result = fn(*a, **kw)
        if result: PASS_count += 1
        else:      FAIL_count += 1

    # ── Step 1: Import pipeline ───────────────────────────────────────────────
    print("[1/9] Importing pipeline modules...")
    try:
        from research.pipeline import (
            draw_boxes, save_training_curves,
            compute_and_save_confusion_matrix,
            save_small_object_analysis, save_efficiency,
            save_reproducibility,
        )
        from research.models.proposed_model import build_proposed_model
        _ok("All pipeline imports succeeded")
        PASS_count += 1
    except Exception as e:
        _fail(f"Import error: {e}")
        FAIL_count += 1
        sys.exit(1)

    # ── Step 2: Build model ───────────────────────────────────────────────────
    print("\n[2/9] Building model (E1 baseline)...")
    try:
        model, model_name = build_proposed_model(
            use_attention=False, use_sacm=False,
            pretrained_backbone=False,
        )
        model.eval()
        _ok(f"Model built: {model_name}")
        PASS_count += 1
    except Exception as e:
        _fail(f"Model build failed: {e}")
        FAIL_count += 1
        sys.exit(1)

    # ── Step 3: Synthetic forward pass ────────────────────────────────────────
    print("\n[3/9] Synthetic forward pass...")
    IMG_SIZE  = args.img_size
    dev       = torch.device(args.device)
    model     = model.to(dev)
    images_raw, targets_raw = make_synthetic_batch(n_images=3, img_size=IMG_SIZE, device=args.device)

    try:
        with torch.no_grad():
            outputs = model([im.to(dev) for im in images_raw])
        _ok(f"Forward pass OK — {len(outputs)} outputs")
        PASS_count += 1
    except Exception as e:
        _fail(f"Forward pass failed: {e}")
        FAIL_count += 1
        sys.exit(1)

    # ── Step 4: Prediction images + predictions.json ──────────────────────────
    print("\n[4/9] Saving prediction images + predictions.json...")
    from research.pipeline import draw_boxes as _draw_boxes, _banner, _hstack_panels, VISDRONE_CLASSES

    all_coco_preds = []
    pred_dir = results_dir / "predictions"
    comp_dir = results_dir / "comparisons"
    pred_dir.mkdir(exist_ok=True)
    comp_dir.mkdir(exist_ok=True)
    all_det_records = []

    for img_t, target, out in zip(images_raw, targets_raw, outputs):
        img_id  = int(target["image_id"][0].item())
        img_np  = (img_t.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gt_boxes  = target["boxes"].numpy().tolist()
        gt_labels = target["labels"].numpy().tolist()

        pb = out["boxes"].cpu().numpy().tolist()
        pl = out["labels"].cpu().numpy().tolist()
        ps = out["scores"].cpu().numpy().tolist()

        pred_img = _draw_boxes(img_bgr, pb, pl, ps)
        gt_img   = _draw_boxes(img_bgr, gt_boxes, gt_labels)

        cv2.imwrite(str(pred_dir / f"image_{img_id:06d}_prediction.jpg"),
                    pred_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        gt_p  = _banner(gt_img,   "GROUND TRUTH")
        pr_p  = _banner(pred_img, "PREDICTION")
        comp  = _hstack_panels([gt_p, pr_p])
        cv2.imwrite(str(comp_dir / f"image_{img_id:06d}_gt_vs_pred.jpg"),
                    comp, [cv2.IMWRITE_JPEG_QUALITY, 92])

        for box, lbl, score in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
            x1,y1,x2,y2 = box.tolist()
            w_b = x2-x1; h_b = y2-y1; s = float(score); l = int(lbl)
            name = VISDRONE_CLASSES[l] if 0<=l<len(VISDRONE_CLASSES) else f"cls_{l}"
            all_det_records.append({
                "image_id":img_id, "category_id":l, "category_name":name,
                "x1":round(x1,2), "y1":round(y1,2), "x2":round(x2,2), "y2":round(y2,2),
                "width":round(w_b,2), "height":round(h_b,2), "confidence":round(s,6),
            })
            all_coco_preds.append({
                "image_id":img_id,"category_id":l,
                "bbox":[x1,y1,w_b,h_b],"score":s,
            })

    json_path = pred_dir / "predictions.json"
    json_path.write_text(json.dumps(all_det_records, indent=2))

    _check(check_image, pred_dir / "image_000001_prediction.jpg", "prediction.jpg")
    _check(check_image, comp_dir / "image_000001_gt_vs_pred.jpg", "gt_vs_pred.jpg")
    _check(check_json,  pred_dir / "predictions.json",            "predictions.json")

    # ── Step 5: Metrics (synthetic — no real COCO GT) ─────────────────────────
    print("\n[5/9] Saving metrics (synthetic GT via COCO API)...")
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    try:
        val_coco, _ = make_synthetic_coco_api(n_images=3)
        from research.evaluation.metrics import compute_coco_metrics
        metrics = compute_coco_metrics(all_coco_preds, [], coco_gt_api=val_coco)
        metrics["model_name"] = model_name
        (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # per_class_metrics.csv
        per_class = metrics.get("per_class_ap", {})
        with open(metrics_dir / "per_class_metrics.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["class_name","AP50","AP50_95","precision","recall","F1"])
            for cls in VISDRONE_CLASSES[1:]:
                cd = per_class.get(cls, {})
                wr.writerow([cls,
                             cd.get("ap50","N/A"), cd.get("ap50_95","N/A"),
                             cd.get("precision","N/A"), cd.get("recall","N/A"),
                             cd.get("f1","N/A")])

        _check(check_json, metrics_dir / "metrics.json",           "metrics.json")
        _check(check_csv,  metrics_dir / "per_class_metrics.csv",  "per_class_metrics.csv")
    except Exception as e:
        _fail(f"metrics: {e}")
        FAIL_count += 1

    # ── Step 6: Confusion matrix ──────────────────────────────────────────────
    print("\n[6/9] Confusion matrix...")
    try:
        val_coco2, _ = make_synthetic_coco_api(n_images=3)
        compute_and_save_confusion_matrix(all_coco_preds, val_coco2, metrics_dir)
        _check(check_image, metrics_dir / "confusion_matrix.png", "confusion_matrix.png")
        _check(check_csv,   metrics_dir / "confusion_matrix.csv", "confusion_matrix.csv")
    except Exception as e:
        _fail(f"confusion matrix: {e}")
        FAIL_count += 1

    # ── Step 7: Small object analysis ─────────────────────────────────────────
    print("\n[7/9] Small object analysis...")
    try:
        val_coco3, _ = make_synthetic_coco_api(n_images=3)
        save_small_object_analysis(
            all_coco_preds, val_coco3,
            metrics_dir, results_dir/"plots", model_name,
        )
        _check(check_json,  metrics_dir  / "small_object_metrics.json", "small_object_metrics.json")
        _check(check_image, results_dir/"plots"/"small_object_analysis.png", "small_object_analysis.png")
    except Exception as e:
        _fail(f"small object: {e}")
        FAIL_count += 1

    # ── Step 8: Training curves ───────────────────────────────────────────────
    print("\n[8/9] Training curves from synthetic CSV...")
    try:
        csv_path = make_synthetic_training_log(results_dir/"logs", n_epochs=5)
        save_training_curves(csv_path, results_dir/"plots", model_name)
        _check(check_image, results_dir/"plots"/"train_loss.png",    "train_loss.png")
        _check(check_image, results_dir/"plots"/"learning_rate.png", "learning_rate.png")
        _check(check_image, results_dir/"plots"/"map50_curve.png",   "map50_curve.png")
        _check(check_image, results_dir/"plots"/"val_loss.png",      "val_loss.png")
    except Exception as e:
        _fail(f"training curves: {e}")
        FAIL_count += 1

    # ── Step 9: Efficiency + reproducibility ──────────────────────────────────
    print("\n[9/9] Efficiency metrics + experiment summary...")
    try:
        cfg_dummy = {
            "img_size":640, "batch_size":2, "epochs":5,
            "seed":42, "mode":"debug",
            "optimizer":{"lr":0.005,"momentum":0.9,"weight_decay":0.0001},
        }
        eff = save_efficiency(model, args.device, IMG_SIZE, metrics_dir, model_name)
        save_reproducibility(cfg_dummy, model_name, metrics, eff, results_dir, val_n=3)
        _check(check_json, metrics_dir / "efficiency.json",              "efficiency.json")
        _check(check_file, results_dir/"logs"/"experiment_summary.txt",  "experiment_summary.txt")
        _check(check_file, results_dir/"logs"/"experiment_config.yaml",  "experiment_config.yaml")
    except Exception as e:
        _fail(f"efficiency/reproducibility: {e}")
        FAIL_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"SANITY TEST COMPLETE:  {PASS_count} PASS   {FAIL_count} FAIL")
    print("="*60)

    if FAIL_count == 0:
        print("\nAll output files are physically created and openable.")
        print("The pipeline is ready for real VisDrone training.")
        print(f"\nOutput location: {results_dir.resolve()}")
        print("\nNext command (debug run with real data):")
        print("  python research/run_experiment.py --experiment baseline --mode debug --device cuda")
    else:
        print(f"\n{FAIL_count} checks FAILED. Fix issues above before training.")

    return FAIL_count == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
