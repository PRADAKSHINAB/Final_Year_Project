"""
research/run_all_experiments.py
================================
Sequential Orchestration Script for E1, E2, E3, and E4 Experiments.

Workflow:
  1. Environment & Dependency Validation
  2. Dataset Validation & COCO JSON Verification
  3. Sequential Training & Post-Training Evaluation for E1 -> E2 -> E3 -> E4
  4. Cross-Experiment Evaluation & Qualitative Grid Generation
  5. Ablation Table, Comparison Figures, and Reproducibility Aggregation

Usage:
  python research/run_all_experiments.py --mode debug --device cuda
  python research/run_all_experiments.py --mode research --device cuda
  python research/run_all_experiments.py --mode research --device cuda --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.check_dependencies import check as check_deps
from research.prepare_dataset import validate_coco_json
from research.run_experiment import EXPERIMENT_CONFIGS, ResultsDirTrainer
from research.models.proposed_model import build_proposed_model, get_model_complexity
from research.datasets.visdrone import build_dataloaders
from research.pipeline import run_pipeline
from research.evaluate_all import (
    save_ablation_table,
    save_publication_figures,
    generate_qualitative_comparison,
    RESULTS_ROOT,
    _load_experiment,
)

EXPERIMENTS_ORDER = ["baseline", "attention", "second_module", "final_model"]


def _parse():
    p = argparse.ArgumentParser(
        description="Run all four SOA-FasterRCNN research experiments sequentially",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",  default="research/configs/fasterrcnn_visdrone.yaml")
    p.add_argument("--mode",    default="research", choices=["debug", "research"])
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",  action="store_true", help="Skip experiments whose best.pth checkpoint exists")
    p.add_argument("--num_vis", type=int, default=50)
    p.add_argument("--conf",    type=float, default=0.25)
    return p.parse_args()


def main():
    args = _parse()
    t_start = time.time()

    print("\n" + "=" * 80)
    print("AUTOMATED SEQUENTIAL EXPERIMENT RUNNER — E1 -> E2 -> E3 -> E4")
    print("=" * 80)

    # 1. Environment & Dependency Check
    print("\n[STEP 1/5] Validating Dependencies...")
    try:
        check_deps()
    except SystemExit as e:
        if e.code != 0:
            print("[ERROR] Dependency check failed. Exiting.")
            sys.exit(1)

    # 2. Dataset Check
    print("\n[STEP 2/5] Validating VisDrone COCO JSON Datasets...")
    if not validate_coco_json(PROJECT_ROOT):
        print("\n[WARNING] COCO JSON files not found or invalid.")
        print("Attempting automatic conversion with research/prepare_dataset.py...")
        from research.prepare_dataset import convert_all_splits
        convert_all_splits(PROJECT_ROOT)
        if not validate_coco_json(PROJECT_ROOT):
            print("[ERROR] Dataset preparation failed. Exiting.")
            sys.exit(1)

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mode_cfg = cfg[args.mode]
    cfg["mode"]        = args.mode
    cfg["device"]      = args.device
    cfg["epochs"]      = mode_cfg["epochs"]
    cfg["img_size"]    = mode_cfg["img_size"]
    cfg["batch_size"]  = mode_cfg["batch_size"]
    cfg["workers"]     = mode_cfg["workers"]
    cfg["amp"]         = cfg.get("training", {}).get("amp", True)
    cfg["grad_clip"]   = cfg.get("training", {}).get("grad_clip", 10.0)

    # Build DataLoaders ONCE for evaluation
    train_root = Path(cfg["dataset"]["train_root"])
    val_root   = Path(cfg["dataset"]["val_root"])
    subset_frac = mode_cfg.get("subset_fraction", 1.0)

    train_loader, val_loader, val_coco = build_dataloaders(
        dataset_root=val_root,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        subset_fraction=subset_frac,
        seed=cfg.get("seed", 42),
        train_root=train_root,
        val_root=val_root,
    )

    all_metrics = []
    models_loaded = []

    # 3. Run Experiments
    print("\n[STEP 3/5] Running Experiments...")
    for idx, exp_key in enumerate(EXPERIMENTS_ORDER, 1):
        exp_cfg = EXPERIMENT_CONFIGS[exp_key]
        subdir  = exp_cfg["results_subdir"]
        exp_dir = RESULTS_ROOT / subdir
        best_ckpt = exp_dir / "checkpoints" / "best.pth"

        print(f"\n" + "-" * 70)
        print(f"[{idx}/4] Experiment: {exp_key.upper()} ({exp_cfg['description']})")
        print(f"Directory:  {exp_dir}")
        print("-" * 70)

        model, model_name = build_proposed_model(
            use_attention=exp_cfg["use_attention"],
            use_sacm=exp_cfg["use_sacm"],
        )

        if args.resume and best_ckpt.exists():
            print(f"[RESUME] Checkpoint found at {best_ckpt}. Skipping training.")
        else:
            print(f"[TRAIN] Starting training ({cfg['epochs']} epochs, size={cfg['img_size']}, batch={cfg['batch_size']})...")
            legacy_dir = Path("experiments") / exp_key
            trainer = ResultsDirTrainer(
                results_dir=exp_dir,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_coco=val_coco,
                cfg=cfg,
                experiment_dir=legacy_dir,
                model_name=model_name,
            )
            trainer.train()

        # Evaluate checkpoint
        if best_ckpt.exists():
            print(f"[EVAL] Evaluating best checkpoint: {best_ckpt}")
            ckpt = torch.load(best_ckpt, map_location=args.device)
            model.load_state_dict(ckpt.get("model", ckpt))

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

            comp = get_model_complexity(model, cfg["img_size"])
            metrics["exp_name"]   = subdir
            metrics["exp_label"]  = exp_key
            metrics["model_name"] = model_name
            metrics["params_M"]   = comp.get("params_M")
            metrics["gflops"]     = comp.get("gflops")

            all_metrics.append(metrics)
            models_loaded.append({"label": exp_key, "model": model.to(args.device), "name": subdir})
        else:
            print(f"[WARNING] Checkpoint {best_ckpt} not found. Skipping evaluation for {exp_key}.")

    # 4. Cross-Experiment Comparison & Figures
    print("\n[STEP 4/5] Generating Cross-Experiment Comparison & Publication Figures...")
    if all_metrics:
        save_ablation_table(all_metrics, RESULTS_ROOT / "ablation_table.csv")
        save_publication_figures(all_metrics, RESULTS_ROOT)

    if len(models_loaded) > 1:
        print("[QUAL] Generating multi-model qualitative comparison grid...")
        generate_qualitative_comparison(
            models_loaded,
            val_loader,
            RESULTS_ROOT / "qualitative_comparison",
            torch.device(args.device),
            cfg["img_size"],
            num_images=min(20, args.num_vis),
            conf_thresh=args.conf,
        )

    # 5. Reproducibility & Summary
    print("\n[STEP 5/5] Writing Overall Reproducibility Record...")
    rep_summary = {
        "experiments_completed": len(all_metrics),
        "total_elapsed_minutes": round((time.time() - t_start) / 60, 2),
        "mode": args.mode,
        "device": args.device,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open(RESULTS_ROOT / "reproducibility.json", "w") as f:
        yaml.dump(rep_summary, f)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Total time elapsed: {rep_summary['total_elapsed_minutes']} minutes")
    print(f"Results stored in:  {RESULTS_ROOT.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
