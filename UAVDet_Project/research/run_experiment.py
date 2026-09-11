"""
research/run_experiment.py
==========================
Main training + evaluation entry point.

Usage:
    # Debug (5% data, 5 epochs, 640px) — ALWAYS RUN FIRST
    python research/run_experiment.py --experiment baseline      --mode debug  --device cuda
    python research/run_experiment.py --experiment attention     --mode debug  --device cuda
    python research/run_experiment.py --experiment second_module --mode debug  --device cuda
    python research/run_experiment.py --experiment final_model   --mode debug  --device cuda

    # Full training (100 epochs, 1280px, full dataset)
    python research/run_experiment.py --experiment baseline      --mode research --device cuda
    python research/run_experiment.py --experiment attention     --mode research --device cuda
    python research/run_experiment.py --experiment second_module --mode research --device cuda
    python research/run_experiment.py --experiment final_model   --mode research --device cuda

    # Evaluation only (run pipeline on best.pth)
    python research/run_experiment.py --experiment baseline --eval-only --device cuda

Output written to:
    research/results/E1_baseline/
    research/results/E2_attention/
    research/results/E3_sacm/
    research/results/E4_full/

DO NOT MODIFY model files, trainer, dataset loader, or config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.datasets.visdrone       import build_dataloaders
from research.models.proposed_model   import build_proposed_model, get_model_complexity
from research.models.baseline_fasterrcnn import load_checkpoint
from research.training.trainer        import ResearchTrainer
from research.pipeline                import run_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENT_CONFIGS = {
    "baseline": {
        "use_attention":  False,
        "use_sacm":       False,
        "results_subdir": "E1_baseline",
        "description":    "E1: Baseline Faster R-CNN + ResNet50-FPN",
    },
    "attention": {
        "use_attention":  True,
        "use_sacm":       False,
        "results_subdir": "E2_attention",
        "description":    "E2: +Coordinate Attention at FPN output P2/P3",
    },
    "second_module": {
        "use_attention":  False,
        "use_sacm":       True,
        "results_subdir": "E3_sacm",
        "description":    "E3: +Scale-Aware Context Module in RPN",
    },
    "final_model": {
        "use_attention":  True,
        "use_sacm":       True,
        "results_subdir": "E4_full",
        "description":    "E4: Full SOA-FasterRCNN (CA + SACM)",
    },
}

RESULTS_ROOT = Path("research/results")


# ──────────────────────────────────────────────────────────────────────────────
# Trainer subclass that writes checkpoints to research/results/{EXP}/checkpoints/
# ──────────────────────────────────────────────────────────────────────────────

class ResultsDirTrainer(ResearchTrainer):
    """
    Thin wrapper around ResearchTrainer.
    Overrides save_checkpoint to write best.pth / last.pth
    under research/results/{EXP}/checkpoints/.
    Also redirects training_log.csv to research/results/{EXP}/logs/.
    The original ResearchTrainer code is NOT modified.
    """

    def __init__(self, results_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_dir = results_dir
        self._ckpt_dir   = results_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Redirect CSV to results/logs/
        self.csv_path = logs_dir / "training_log.csv"
        self._init_csv()

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        state = {
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "metrics":    metrics,
            "model_name": self.model_name,
        }
        torch.save(state, self._ckpt_dir / "last.pth")
        if is_best:
            torch.save(state, self._ckpt_dir / "best.pth")
            m50 = metrics.get("map50", 0)
            print(f"  [BEST] mAP@50={m50:.4f} at epoch {epoch} -> checkpoints/best.pth")

        # Keep legacy copy in experiment_dir/weights/ (trainer compatibility)
        wdir = self.experiment_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        torch.save(state, wdir / "last_model.pth")
        if is_best:
            torch.save(state, wdir / "best_model.pth")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="SOA-FasterRCNN experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS))
    p.add_argument("--config",  default="research/configs/fasterrcnn_visdrone.yaml")
    p.add_argument("--mode",    default="research", choices=["debug","research"],
                   help="'debug' = 5%% data, 5 epochs | 'research' = full")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; run evaluation pipeline on best.pth")
    p.add_argument("--checkpoint", default=None,
                   help="Custom checkpoint path for eval-only mode")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--num_vis", type=int, default=50,
                   help="Number of prediction images to save")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold for visualisations")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args    = _parse()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["mode"]   = args.mode
    cfg["device"] = args.device
    cfg["seed"]   = args.seed

    mode_cfg = cfg[args.mode]
    cfg["epochs"]     = mode_cfg["epochs"]
    cfg["img_size"]   = mode_cfg["img_size"]
    cfg["batch_size"] = mode_cfg["batch_size"]
    cfg["workers"]    = mode_cfg["workers"]

    # Flatten nested config keys that trainer expects
    cfg["amp"]        = cfg.get("training", {}).get("amp", True)
    cfg["grad_clip"]  = cfg.get("training", {}).get("grad_clip", 10.0)
    cfg["log_interval"] = cfg.get("training", {}).get("log_interval", 50)
    cfg["freeze_backbone_epochs"] = cfg.get("model", {}).get("freeze_backbone_epochs", 5)
    cfg["early_stopping_patience"] = cfg.get("training", {}).get("early_stopping_patience", 20)

    exp_cfg     = EXPERIMENT_CONFIGS[args.experiment]
    results_dir = RESULTS_ROOT / exp_cfg["results_subdir"]

    # Create standardized output structure
    for sub in ["checkpoints","predictions","comparisons","plots","metrics","logs"]:
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    # Legacy dir for trainer internals
    legacy_dir = Path(cfg.get("output",{}).get("experiment_dir","experiments")) / args.experiment
    legacy_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Experiment : {args.experiment}  ({exp_cfg['description']})")
    print(f"Mode       : {args.mode}")
    print(f"Device     : {args.device}")
    print(f"Results    : {results_dir.resolve()}")
    print("=" * 70)

    # Save config copy
    with open(results_dir / "logs" / "experiment_config.yaml", "w") as f:
        yaml.dump({**cfg, "experiment": args.experiment}, f)

    # ── Build model ───────────────────────────────────────────────────────────
    model, model_name = build_proposed_model(
        use_attention=exp_cfg["use_attention"],
        use_sacm=exp_cfg["use_sacm"],
        attention_type=cfg.get("attention",{}).get("type","coordinate_attention"),
        attention_location=cfg.get("attention",{}).get("location","fpn_lateral"),
        attention_reduction=cfg.get("attention",{}).get("reduction_ratio",32),
        sacm_n_bins=cfg.get("scale_aware_rpn",{}).get("scale_bins",4),
        sacm_context_channels=cfg.get("scale_aware_rpn",{}).get("context_channels",64),
    )
    comp = get_model_complexity(model, cfg["img_size"])
    print(f"Model      : {model_name}  |  {comp['params_M']}M params  |  GFLOPs={comp['gflops']}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_root      = Path(cfg.get("dataset", {}).get("train_root", cfg["dataset"]["root"]))
    val_root        = Path(cfg.get("dataset", {}).get("val_root", cfg["dataset"]["root"]))
    subset_fraction = mode_cfg.get("subset_fraction", 1.0) if args.mode == "debug" else 1.0

    train_loader, val_loader, val_coco = build_dataloaders(
        dataset_root=val_root,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        subset_fraction=subset_fraction,
        seed=args.seed,
        train_root=train_root,
        val_root=val_root,
    )
    print(f"Train      : {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}")

    # ── Eval-only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        ckpt = args.checkpoint or str(results_dir / "checkpoints" / "best.pth")
        if not Path(ckpt).exists():
            ckpt = str(legacy_dir / "weights" / "best_model.pth")
        print(f"[EVAL-ONLY] Loading: {ckpt}")
        load_checkpoint(model, ckpt, device=args.device)
        run_pipeline(
            model=model, val_loader=val_loader, val_coco=val_coco,
            results_dir=results_dir, model_name=model_name,
            cfg=cfg, device=args.device,
            num_vis=args.num_vis, conf_thresh=args.conf,
        )
        return

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = ResultsDirTrainer(
        results_dir=results_dir,
        model=model, train_loader=train_loader, val_loader=val_loader,
        val_coco=val_coco, cfg=cfg,
        experiment_dir=legacy_dir, model_name=model_name,
    )
    trainer.train()

    # ── Post-training evaluation pipeline ─────────────────────────────────────
    best_ckpt = results_dir / "checkpoints" / "best.pth"
    if not best_ckpt.exists():
        best_ckpt = legacy_dir / "weights" / "best_model.pth"

    if best_ckpt.exists():
        print(f"\n[POST-TRAIN] Loading best checkpoint: {best_ckpt}")
        load_checkpoint(model, str(best_ckpt), device=args.device)
        run_pipeline(
            model=model, val_loader=val_loader, val_coco=val_coco,
            results_dir=results_dir, model_name=model_name,
            cfg=cfg, device=args.device,
            num_vis=args.num_vis, conf_thresh=args.conf,
        )
    else:
        print("[WARNING] No best checkpoint found; skipping post-training pipeline.")

    print(f"\n{'='*70}")
    print(f"[DONE]  Experiment: {args.experiment}")
    print(f"  checkpoints/ : {results_dir/'checkpoints'}")
    print(f"  predictions/ : {results_dir/'predictions'}")
    print(f"  comparisons/ : {results_dir/'comparisons'}")
    print(f"  metrics/     : {results_dir/'metrics'}")
    print(f"  plots/       : {results_dir/'plots'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
