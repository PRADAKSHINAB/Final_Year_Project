"""
Research Training Script — Thin CLI wrapper around ResearchTrainer.

This script is called by run_experiment.py but can also be used directly.
Usage:
    python research/training/train.py \
        --config research/configs/fasterrcnn_visdrone.yaml \
        --experiment baseline \
        --mode research \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.datasets.visdrone import build_dataloaders
from research.models.proposed_model import build_proposed_model
from research.training.trainer import ResearchTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="research/configs/fasterrcnn_visdrone.yaml")
    p.add_argument("--experiment", default="baseline",
                   choices=["baseline", "attention", "second_module", "final_model"])
    p.add_argument("--mode", default="research", choices=["debug", "research"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


EXPERIMENT_TO_MODEL_FLAGS = {
    "baseline": (False, False),
    "attention": (True, False),
    "second_module": (False, True),
    "final_model": (True, True),
}


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["mode"] = args.mode
    cfg["device"] = args.device
    cfg["seed"] = args.seed

    mode_cfg = cfg[args.mode]
    cfg["epochs"] = mode_cfg["epochs"]
    cfg["img_size"] = mode_cfg["img_size"]
    cfg["batch_size"] = mode_cfg["batch_size"]
    cfg["workers"] = mode_cfg["workers"]

    use_attention, use_sacm = EXPERIMENT_TO_MODEL_FLAGS[args.experiment]
    model, model_name = build_proposed_model(
        use_attention=use_attention,
        use_sacm=use_sacm,
    )

    experiment_dir = Path(cfg["output"]["experiment_dir"]) / args.experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(cfg["dataset"]["root"])
    subset = mode_cfg.get("subset_fraction", 1.0) if args.mode == "debug" else 1.0

    train_loader, val_loader, val_coco = build_dataloaders(
        dataset_root=dataset_root,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        subset_fraction=subset,
        seed=cfg["seed"],
    )

    trainer = ResearchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_coco=val_coco,
        cfg=cfg,
        experiment_dir=experiment_dir,
        model_name=model_name,
    )
    trainer.train()


if __name__ == "__main__":
    main()
