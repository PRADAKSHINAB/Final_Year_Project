"""
research/evaluate_all.py
=========================
Evaluate all four experiments (E1-E4) on the SAME validation set.

Produces:
  research/results/E{1..4}_*/metrics/metrics.json
  research/results/E{1..4}_*/predictions/*.jpg
  research/results/E{1..4}_*/comparisons/*_gt_vs_pred.jpg
  research/results/qualitative_comparison/comparison_*.jpg
  research/results/ablation_table.csv
  research/results/final_comparison.csv
  research/results/ablation_map.png
  research/results/ablation_ap_small.png
  research/results/ablation_fps.png
  research/results/model_comparison.png

Usage:
    python research/evaluate_all.py --device cuda
    python research/evaluate_all.py --device cuda --mode debug --num_vis 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.datasets.visdrone import build_dataloaders
from research.models.proposed_model import build_proposed_model, get_model_complexity
from research.models.baseline_fasterrcnn import load_checkpoint
from research.pipeline import (
    run_pipeline,
    VISDRONE_CLASSES,
    _COLORS_BGR,
    _cls_name,
    draw_boxes,
    _banner,
    _hstack_panels,
)

# ──────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "key":          "baseline",
        "name":         "E1_baseline",
        "label":        "E1 Baseline",
        "use_attention": False,
        "use_sacm":      False,
    },
    {
        "key":          "attention",
        "name":         "E2_attention",
        "label":        "E2 +CA",
        "use_attention": True,
        "use_sacm":      False,
    },
    {
        "key":          "second_module",
        "name":         "E3_sacm",
        "label":        "E3 +SACM",
        "use_attention": False,
        "use_sacm":      True,
    },
    {
        "key":          "final_model",
        "name":         "E4_full",
        "label":        "E4 Full",
        "use_attention": True,
        "use_sacm":      True,
    },
]

RESULTS_ROOT = Path("research/results")


# ──────────────────────────────────────────────────────────────────────────────
# Load one experiment's model
# ──────────────────────────────────────────────────────────────────────────────

def _load_experiment(exp: Dict, results_root: Path, device: torch.device):
    """Return (model, model_name) or None if checkpoint missing."""
    ckpt = results_root / exp["name"] / "checkpoints" / "best.pth"
    if not ckpt.exists():
        print(f"  [SKIP] {exp['name']}: checkpoint not found at {ckpt}")
        return None, None

    model, model_name = build_proposed_model(
        use_attention=exp["use_attention"],
        use_sacm=exp["use_sacm"],
        pretrained_backbone=False,
    )
    load_checkpoint(model, str(ckpt), device=str(device))
    return model.to(device).eval(), model_name


# ──────────────────────────────────────────────────────────────────────────────
# Qualitative comparison: Original | GT | E1 | E2 | E3 | E4
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_qualitative_comparison(
    models_loaded: List[Dict],
    val_loader,
    output_dir: Path,
    device: torch.device,
    img_size: int,
    num_images: int = 20,
    conf_thresh: float = 0.25,
) -> None:
    """Save Original|GT|E1|E2|E3|E4 side-by-side images using identical input images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for images, targets in val_loader:
        if saved >= num_images:
            break

        for img_t, target in zip(images, targets):
            if saved >= num_images:
                break

            img_id   = int(target["image_id"][0].item())
            img_np   = (img_t.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
            img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gt_boxes  = target["boxes"].cpu().numpy().tolist()
            gt_labels = target["labels"].cpu().numpy().tolist()

            panels  = [_banner(img_bgr.copy(), "ORIGINAL")]
            gt_img  = draw_boxes(img_bgr, gt_boxes, gt_labels)
            panels.append(_banner(gt_img, "GROUND TRUTH"))

            for exp_info in models_loaded:
                mdl = exp_info["model"]
                out = mdl([img_t.to(device)])[0]
                keep   = out["scores"].cpu() >= conf_thresh
                pb     = out["boxes"].cpu()[keep].numpy().tolist()
                pl     = out["labels"].cpu()[keep].numpy().tolist()
                ps     = out["scores"].cpu()[keep].numpy().tolist()
                p_img  = draw_boxes(img_bgr, pb, pl, ps)
                panels.append(_banner(p_img, exp_info["label"]))

            combined = _hstack_panels(panels)
            out_path = output_dir / f"comparison_{img_id:04d}.jpg"
            cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

    print(f"[QUAL] {saved} qualitative comparison images -> {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Ablation CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_ablation_table(all_metrics: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Experiment", "mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large",
        "Precision", "Recall", "F1", "FPS", "Parameters", "GFLOPs"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_metrics:
            def _g(k):
                v = m.get(k)
                if v is None: return "N/A"
                return round(v, 4) if isinstance(v, float) else v
            w.writerow({
                "Experiment": m.get("exp_label", m.get("exp_name", "")),
                "mAP50":      _g("map50"),
                "mAP50_95":   _g("map50_95"),
                "AP_small":   _g("ap_small"),
                "AP_medium":  _g("ap_medium"),
                "AP_large":   _g("ap_large"),
                "Precision":  _g("precision"),
                "Recall":     _g("recall"),
                "F1":         _g("f1"),
                "FPS":        _g("fps"),
                "Parameters": m.get("params_M", "N/A"),
                "GFLOPs":     m.get("gflops", "N/A"),
            })
    print(f"[TABLE] ablation_table.csv -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Publication figures
# ──────────────────────────────────────────────────────────────────────────────

def save_publication_figures(all_metrics: List[Dict], results_root: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available for publication figures.")
        return

    ok = [m for m in all_metrics if m.get("map50") is not None]
    if not ok:
        print("[WARNING] No completed experiments to plot publication figures.")
        return

    labels = [m.get("exp_label", m.get("exp_name","?")) for m in ok]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(ok)]
    x = np.arange(len(ok))

    def _bar(vals, title, ylabel, fname, fmt=".4f"):
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(x, [v if v is not None else 0 for v in vals],
                      color=colors, edgecolor="black", linewidth=0.5, width=0.55)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v and v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{v:{fmt}}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(results_root / fname, dpi=200, bbox_inches="tight")
        plt.close()

    # Requested publication figure filenames:
    # 1. ablation_map.png
    _bar([m.get("map50") for m in ok], "mAP@0.50 Ablation Comparison", "mAP@0.50", "ablation_map.png")
    
    # 2. ablation_ap_small.png
    _bar([m.get("ap_small") for m in ok], "AP_small Ablation Comparison", "AP_small", "ablation_ap_small.png")
    
    # 3. ablation_fps.png
    _bar([m.get("fps") for m in ok], "Inference FPS Ablation Comparison", "FPS", "ablation_fps.png", fmt=".1f")

    # 4. model_comparison.png (Multi-metric grouped bar)
    metric_keys  = ["map50", "map50_95", "ap_small", "ap_medium", "ap_large"]
    metric_lbls  = ["mAP@0.5", "mAP@0.5:0.95", "AP_small", "AP_medium", "AP_large"]
    fig, ax = plt.subplots(figsize=(12, 5))
    n_m = len(metric_keys); w = 0.14
    for mi, (mk, ml) in enumerate(zip(metric_keys, metric_lbls)):
        offs = x - (n_m - 1) * w / 2 + mi * w
        vals = [m.get(mk, 0) or 0 for m in ok]
        ax.bar(offs, vals, w, label=ml)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=8, ha="right", fontsize=9)
    ax.set_ylabel("AP / mAP", fontsize=10)
    ax.set_title("Full Model Comparison — E1 vs E2 vs E3 vs E4", fontsize=12)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_root / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"[PLOTS] Publication figures (ablation_map.png, ablation_ap_small.png, ablation_fps.png, model_comparison.png) -> {results_root}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Evaluate all four SOA-FasterRCNN experiments")
    p.add_argument("--config",  default="research/configs/fasterrcnn_visdrone.yaml")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mode",    default="research", choices=["debug","research"])
    p.add_argument("--num_vis", type=int, default=20, help="Images to visualize per experiment")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--skip_qual", action="store_true", help="Skip qualitative comparison")
    return p.parse_args()


def main():
    args = _parse()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mode_cfg = cfg[args.mode]
    cfg.update({
        "img_size":   mode_cfg["img_size"],
        "batch_size": mode_cfg["batch_size"],
        "workers":    mode_cfg["workers"],
    })

    dev = torch.device(args.device)
    img_size = cfg["img_size"]

    _, val_loader, val_coco = build_dataloaders(
        dataset_root=Path(cfg["dataset"]["root"]),
        img_size=img_size,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        subset_fraction=1.0,
        seed=cfg.get("seed", 42),
    )
    print(f"\nValidation set: {len(val_loader.dataset)} images")
    print(f"{'='*70}")

    all_metrics: List[Dict] = []
    models_loaded: List[Dict] = []

    for exp in EXPERIMENTS:
        print(f"\n--- {exp['name']} ({exp['label']}) ---")
        exp_dir = RESULTS_ROOT / exp["name"]

        model, model_name = _load_experiment(exp, RESULTS_ROOT, dev)
        if model is None:
            all_metrics.append({
                "exp_name": exp["name"], "exp_label": exp["label"],
                "status": "CHECKPOINT_NOT_FOUND",
            })
            continue

        metrics = run_pipeline(
            model=model,
            val_loader=val_loader,
            val_coco=val_coco,
            results_dir=exp_dir,
            model_name=model_name,
            cfg=cfg,
            device=args.device,
            num_vis=args.num_vis,
            conf_thresh=args.conf,
        )

        comp = get_model_complexity(model, img_size)
        metrics["exp_name"]   = exp["name"]
        metrics["exp_label"]  = exp["label"]
        metrics["model_name"] = model_name
        metrics["params_M"]   = comp.get("params_M")
        metrics["gflops"]     = comp.get("gflops")
        all_metrics.append(metrics)

        models_loaded.append({"label": exp["label"], "model": model, "name": exp["name"]})

    if not args.skip_qual and len(models_loaded) > 1:
        print(f"\n[QUAL] Generating qualitative comparison for {len(models_loaded)} models...")
        generate_qualitative_comparison(
            models_loaded, val_loader,
            RESULTS_ROOT / "qualitative_comparison",
            dev, img_size, num_images=args.num_vis, conf_thresh=args.conf,
        )

    save_ablation_table(all_metrics, RESULTS_ROOT / "ablation_table.csv")
    save_publication_figures(all_metrics, RESULTS_ROOT)

    print(f"\n{'='*78}")
    print(f"{'Experiment':<18} {'mAP@50':>8} {'mAP50:95':>10} {'AP_small':>10} {'FPS':>7} {'Params M':>9}")
    print(f"{'-'*78}")
    for m in all_metrics:
        if m.get("status") == "CHECKPOINT_NOT_FOUND":
            print(f"{m['exp_label']:<18} {'SKIPPED'}")
            continue
        print(f"{m.get('exp_label',''):<18} "
              f"{m.get('map50',0):>8.4f} "
              f"{m.get('map50_95',0):>10.4f} "
              f"{m.get('ap_small',0):>10.4f} "
              f"{m.get('fps',0):>7.1f} "
              f"{m.get('params_M',0):>9.2f}")
    print(f"{'='*78}")
    print(f"\nResults root: {RESULTS_ROOT.resolve()}")


if __name__ == "__main__":
    main()
