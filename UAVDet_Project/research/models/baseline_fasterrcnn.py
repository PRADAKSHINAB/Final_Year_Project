"""
Baseline Faster R-CNN Model Factory.

Implements Faster R-CNN + ResNet-50-FPN as the research baseline.
This is Experiment 1 in the ablation study.

Architecture:
- Backbone: ResNet-50 with ImageNet pretrained weights (IMAGENET1K_V2)
- Neck: Feature Pyramid Network (FPN) with 5 output levels (P2-P6)
- Head: Faster R-CNN V2 box head (4 conv + 1 fc)
- Anchors: Custom small-object scales (16, 32, 64, 128, 256 px)
- Aspect ratios: (0.5, 1.0, 2.0, 3.0) — includes elongated vehicles

Modifications vs. standard torchvision Faster R-CNN:
1. Minimum anchor scale reduced to 16px (default: 32px)
   Justification: VisDrone median object size ~8x10px; need anchors < 32px
2. Extended aspect ratio 3.0 for elongated vehicles (trucks, buses)
3. Higher RPN top-K (train: 4000/2000, test: 6000/1000)
   Justification: Dense UAV scenes produce many small-object proposals
4. Lower box score threshold (0.05) during evaluation for recall
5. Backbone freeze for first N epochs to stabilize RPN training
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# VisDrone: 10 foreground classes + 1 background = 11
VISDRONE_NUM_CLASSES = 11

VISDRONE_CLASS_NAMES = [
    "__background__",
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def build_baseline_model(
    num_classes: int = VISDRONE_NUM_CLASSES,
    pretrained_backbone: bool = True,
    backbone_weights: str = "IMAGENET1K_V2",
    trainable_layers: int = 3,
    returned_layers: Optional[list] = None,
    anchor_sizes: Optional[tuple] = None,
    aspect_ratios: Optional[tuple] = None,
    rpn_pre_nms_top_n_train: int = 4000,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 6000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    box_nms_thresh: float = 0.5,
    box_score_thresh: float = 0.05,
) -> nn.Module:
    """
    Build the baseline Faster R-CNN model for VisDrone.

    Args:
        num_classes: Number of output classes including background.
        pretrained_backbone: Whether to use ImageNet pretrained backbone.
        backbone_weights: Which ImageNet weights to load.
        trainable_layers: Number of backbone layers to unfreeze (after warmup).
        returned_layers: Which ResNet layers to use for FPN.
        anchor_sizes: Anchor sizes per FPN level.
        aspect_ratios: Aspect ratios per FPN level.
        rpn_*: RPN configuration.
        box_*: Box head configuration.

    Returns:
        Faster R-CNN model configured for VisDrone small-object detection.
    """
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]  # C2, C3, C4, C5 -> P2, P3, P4, P5

    if anchor_sizes is None:
        # Start at 16px for VisDrone tiny objects (vs 32px default)
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))

    if aspect_ratios is None:
        # 3.0 added for elongated vehicles; standard 0.5, 1.0, 2.0 preserved
        aspect_ratios = ((0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)

    anchor_gen = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=backbone_weights if pretrained_backbone else None,
        trainable_layers=trainable_layers,
        returned_layers=returned_layers,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_nms_thresh=box_nms_thresh,
        box_score_thresh=box_score_thresh,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
    )

    if pretrained_backbone:
        # Load detection-pretrained weights where architecture matches.
        # This initializes the backbone + FPN from COCO-pretrained weights,
        # while the classification/regression heads remain randomly initialized
        # (they are reinitialized for VisDrone's 10 classes).
        try:
            ref_weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict(progress=True)
            own = model.state_dict()
            filtered = {
                k: v for k, v in ref_weights.items()
                if k in own and own[k].shape == v.shape
            }
            own.update(filtered)
            model.load_state_dict(own)
            print(f"[INFO] Loaded {len(filtered)}/{len(own)} parameters from COCO pretrained weights.")
        except Exception as e:
            print(f"[WARNING] Could not load COCO pretrained weights: {e}. Using ImageNet backbone only.")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters (for first N training epochs)."""
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters (after warmup epochs)."""
    for p in model.backbone.parameters():
        p.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load model weights from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    # Handle partial loads (e.g., after architecture modification)
    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    print(f"[INFO] Loaded {len(matched)}/{len(own)} parameters from {checkpoint_path}")
    return ckpt
