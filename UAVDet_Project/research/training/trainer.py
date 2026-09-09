"""
Research Trainer for Small-Object-Aware Faster R-CNN.

Implements reproducible training with:
- Deterministic seeding
- Mixed precision (AMP)
- Cosine LR with warmup
- Epoch-level validation
- Full metric logging (including AP_small)
- Best model checkpointing
- CSV + JSON logging
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from research.evaluation.metrics import compute_coco_metrics, measure_inference_time


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: full determinism with torch.backends.cudnn.deterministic=True
    # incurs significant performance cost; disabled by default.


@dataclass
class EpochResult:
    """Results from one training epoch."""
    epoch: int
    loss_total: float
    loss_rpn_cls: float
    loss_rpn_box: float
    loss_roi_cls: float
    loss_roi_box: float
    lr: float
    epoch_time_s: float
    # Validation metrics (filled after validation)
    map50: float = 0.0
    map50_95: float = 0.0
    ap_small: float = 0.0
    ap_medium: float = 0.0
    ap_large: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class WarmupCosineScheduler:
    """
    Cosine annealing scheduler with linear warmup.

    During warmup (epochs 0..warmup_epochs-1):
        lr = base_lr * (epoch/warmup_epochs) * warmup_scale
    After warmup:
        lr = cosine annealing from base_lr to eta_min
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        eta_min: float,
        warmup_scale: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.warmup_scale = warmup_scale
        self.current_epoch = 0

    def step(self) -> None:
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            scale = self.warmup_scale + (1.0 - self.warmup_scale) * (
                self.current_epoch / self.warmup_epochs
            )
            lr = self.base_lr * scale
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + np.cos(np.pi * progress)
            )
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class ResearchTrainer:
    """
    Full research trainer with proper logging and checkpointing.

    Usage:
        trainer = ResearchTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_coco=val_coco,
            cfg=cfg,
            experiment_dir=Path("experiments/baseline"),
        )
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_coco: Any,
        cfg: Dict[str, Any],
        experiment_dir: Path,
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_coco = val_coco
        self.cfg = cfg
        self.experiment_dir = Path(experiment_dir)
        self.model_name = model_name

        # Setup
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)

        set_seed(cfg.get("seed", 42))
        self._setup_dirs()
        self._build_optimizer()
        self.scaler = GradScaler(enabled=(self.device.type == "cuda" and cfg.get("amp", True)))

        self.best_map50 = -1.0
        self.epochs_without_improvement = 0

        # Logging
        self.history: List[EpochResult] = []
        self._init_csv()

    def _setup_dirs(self) -> None:
        for subdir in ["weights", "logs", "plots", "predictions"]:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self) -> None:
        opt_cfg = self.cfg.get("optimizer", {})
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SGD(
            trainable,
            lr=opt_cfg.get("lr", 0.005),
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0001),
            nesterov=opt_cfg.get("nesterov", True),
        )

        sch_cfg = self.cfg.get("scheduler", {})
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=sch_cfg.get("warmup_epochs", 3),
            total_epochs=self.cfg.get("epochs", 100),
            base_lr=opt_cfg.get("lr", 0.005),
            eta_min=opt_cfg.get("lr", 0.005) * sch_cfg.get("eta_min_fraction", 0.01),
            warmup_scale=sch_cfg.get("warmup_lr_scale", 0.1),
        )

    def _init_csv(self) -> None:
        self.csv_path = self.experiment_dir / "logs" / "training_log.csv"
        headers = [
            "epoch", "loss_total", "loss_rpn_cls", "loss_rpn_box",
            "loss_roi_cls", "loss_roi_box", "lr", "epoch_time_s",
            "map50", "map50_95", "ap_small", "ap_medium", "ap_large",
            "precision", "recall", "f1",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    def _log_row(self, result: EpochResult) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "loss_total", "loss_rpn_cls", "loss_rpn_box",
                "loss_roi_cls", "loss_roi_box", "lr", "epoch_time_s",
                "map50", "map50_95", "ap_small", "ap_medium", "ap_large",
                "precision", "recall", "f1",
            ])
            writer.writerow({
                "epoch": result.epoch,
                "loss_total": f"{result.loss_total:.6f}",
                "loss_rpn_cls": f"{result.loss_rpn_cls:.6f}",
                "loss_rpn_box": f"{result.loss_rpn_box:.6f}",
                "loss_roi_cls": f"{result.loss_roi_cls:.6f}",
                "loss_roi_box": f"{result.loss_roi_box:.6f}",
                "lr": f"{result.lr:.8f}",
                "epoch_time_s": f"{result.epoch_time_s:.2f}",
                "map50": f"{result.map50:.6f}",
                "map50_95": f"{result.map50_95:.6f}",
                "ap_small": f"{result.ap_small:.6f}",
                "ap_medium": f"{result.ap_medium:.6f}",
                "ap_large": f"{result.ap_large:.6f}",
                "precision": f"{result.precision:.6f}",
                "recall": f"{result.recall:.6f}",
                "f1": f"{result.f1:.6f}",
            })

    def train_epoch(self, epoch: int) -> EpochResult:
        """Train for one epoch and return aggregated losses."""
        self.model.train()
        t0 = time.time()

        # Unfreeze backbone after warmup
        freeze_epochs = self.cfg.get("freeze_backbone_epochs", 5)
        if epoch == freeze_epochs + 1:
            for p in self.model.backbone.parameters():
                p.requires_grad = True
            self._build_optimizer()
            print(f"[Epoch {epoch}] Backbone unfrozen, optimizer reset.")

        running = {}
        n_steps = 0
        log_interval = self.cfg.get("log_interval", 50)
        amp = self.cfg.get("amp", True) and self.device.type == "cuda"

        for step, (images, targets) in enumerate(self.train_loader, 1):
            images = [im.to(self.device, non_blocking=True) for im in images]
            targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast(enabled=amp):
                loss_dict = self.model(images, targets)
                loss = sum(l for l in loss_dict.values())

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip", 10.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            running["total"] = running.get("total", 0.0) + float(loss.detach())
            n_steps += 1

            if step % log_interval == 0:
                avg = {k: round(v / n_steps, 5) for k, v in running.items()}
                lr = self.scheduler.get_lr()
                print(f"[E{epoch:03d} S{step:05d}] lr={lr:.6f} " + " | ".join(f"{k}={v}" for k, v in avg.items()))

        avg = {k: v / max(n_steps, 1) for k, v in running.items()}

        return EpochResult(
            epoch=epoch,
            loss_total=avg.get("total", 0.0),
            loss_rpn_cls=avg.get("loss_objectness", avg.get("loss_rpn_cls", 0.0)),
            loss_rpn_box=avg.get("loss_rpn_box_reg", 0.0),
            loss_roi_cls=avg.get("loss_classifier", 0.0),
            loss_roi_box=avg.get("loss_box_reg", 0.0),
            lr=self.scheduler.get_lr(),
            epoch_time_s=time.time() - t0,
        )

    @torch.inference_mode()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute full COCO metrics."""
        self.model.eval()
        predictions: List[Dict] = []
        img_size = self.cfg.get("img_size", 640)

        for images, targets in self.val_loader:
            images = [im.to(self.device, non_blocking=True) for im in images]
            outputs = self.model(images)

            for target, out in zip(targets, outputs):
                img_id = int(target["image_id"][0].item())
                meta = self.val_coco.loadImgs([img_id])[0]
                orig_w = meta.get("width", img_size)
                orig_h = meta.get("height", img_size)
                scale_x = orig_w / img_size
                scale_y = orig_h / img_size

                for box, label, score in zip(
                    out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
                ):
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(label.item()),
                        "bbox": [
                            x1 * scale_x,
                            y1 * scale_y,
                            (x2 - x1) * scale_x,
                            (y2 - y1) * scale_y,
                        ],
                        "score": float(score.item()),
                    })

        metrics = compute_coco_metrics(predictions, [], coco_gt_api=self.val_coco)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        weights_dir = self.experiment_dir / "weights"
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_name": self.model_name,
        }
        torch.save(state, weights_dir / "last_model.pth")
        if is_best:
            torch.save(state, weights_dir / "best_model.pth")
            print(f"  -> New best mAP@50: {metrics.get('map50', 0):.4f} (saved best_model.pth)")

    def train(self) -> None:
        """Full training loop."""
        epochs = self.cfg.get("epochs", 100)
        patience = self.cfg.get("early_stopping_patience", 20)

        print("=" * 70)
        print(f"Training: {self.model_name}")
        print(f"  Device : {self.device}")
        print(f"  Epochs : {epochs}")
        print(f"  Experiment dir: {self.experiment_dir}")
        print("=" * 70)

        # Freeze backbone initially
        if self.cfg.get("freeze_backbone_epochs", 5) > 0:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
            print(f"[INFO] Backbone frozen for first {self.cfg.get('freeze_backbone_epochs',5)} epochs")

        for epoch in range(1, epochs + 1):
            self.scheduler.step()

            # Train
            result = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()
            result.map50 = val_metrics["map50"]
            result.map50_95 = val_metrics["map50_95"]
            result.ap_small = val_metrics["ap_small"]
            result.ap_medium = val_metrics["ap_medium"]
            result.ap_large = val_metrics["ap_large"]
            result.precision = val_metrics["precision"]
            result.recall = val_metrics["recall"]
            result.f1 = val_metrics["f1"]

            # Log
            self._log_row(result)
            self.history.append(result)

            print(
                f"[E{epoch:03d}] mAP50={result.map50:.4f} mAP50:95={result.map50_95:.4f} "
                f"AP_s={result.ap_small:.4f} AP_m={result.ap_medium:.4f} AP_l={result.ap_large:.4f} "
                f"P={result.precision:.3f} R={result.recall:.3f} F1={result.f1:.3f}"
            )

            # Checkpoint
            is_best = result.map50 > self.best_map50
            if is_best:
                self.best_map50 = result.map50
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Early stopping
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Save final metrics
        final_metrics_path = self.experiment_dir / "metrics.json"
        with open(final_metrics_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "best_map50": self.best_map50,
                "final_epoch": epoch,
                "history_length": len(self.history),
                "note": "Per-epoch results in training_log.csv",
            }, f, indent=2)

        print(f"\n[DONE] Best mAP@50: {self.best_map50:.4f}")
        print(f"       Checkpoints saved to: {self.experiment_dir / 'weights'}")
        print(f"       Log saved to: {self.csv_path}")
