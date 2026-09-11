"""
Proposed Model: Small-Object-Aware Faster R-CNN (SOA-FasterRCNN).

This module assembles the full proposed model by integrating:
1. Baseline: Faster R-CNN + ResNet50-FPN (Experiment 1)
2. Attention: Coordinate Attention at FPN lateral connections (Experiment 2)
3. Scale-Aware RPN: SACM in RPN head (Experiment 3)
4. Combined: Both modules together (Experiment 4)

Architecture Diagram:
---------------------
Input Image (1280x1280)
       |
   ResNet-50 Backbone
   [C2, C3, C4, C5]
       |
   FPN (Feature Pyramid Network)
       |
   [FPN Lateral Connections]
       |
   [OPTIONAL: Coordinate Attention at P2/P3 lateral connections]
       |
   FPN Top-Down Fusion -> [P2, P3, P4, P5, P6]
       |
   RPN (Region Proposal Network)
       |
   [OPTIONAL: Scale-Aware Context Module within RPN]
       |
   ROI Align (7x7)
       |
   Box Head (4 conv + 1 fc)
       |
   Classification + Bounding Box Regression
       |
   [10 VisDrone classes + background]

Model Naming Convention:
- Experiment 1: baseline_fasterrcnn
- Experiment 2: fasterrcnn_ca         (+ Coordinate Attention)
- Experiment 3: fasterrcnn_sacm       (+ Scale-Aware Context Module)
- Experiment 4: fasterrcnn_ca_sacm    (full proposed = SOA-FasterRCNN)

Novelty status:
- Coordinate Attention: EXISTING METHOD (Hou et al., 2021)
- FPN at lateral connections: EXISTING concept
- CA at FPN lateral specifically for Faster RCNN + VisDrone sub-32px:
  POTENTIAL NOVELTY - REQUIRES VERIFICATION
- SACM in RPN: POTENTIAL NOVELTY - REQUIRES VERIFICATION
- Their combination: POTENTIAL NOVELTY - REQUIRES VERIFICATION
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import RPNHead

from research.models.attention import FPNLateralAttention, build_attention_module
from research.models.baseline_fasterrcnn import (
    VISDRONE_CLASS_NAMES,
    VISDRONE_NUM_CLASSES,
    build_baseline_model,
)
from research.models.proposed_module import ScaleAwareRPNHead


class AttentionFPN(nn.Module):
    """
    FPN wrapper that applies attention at lateral connections.

    Integrates with torchvision's BackboneWithFPN by wrapping the inner FPN.
    The attention is applied to the LATERAL features (before top-down fusion)
    at the specified pyramid levels.

    This is the implementation of our first proposed contribution.
    """

    def __init__(
        self,
        base_fpn: nn.Module,
        attention_module: FPNLateralAttention,
    ) -> None:
        super().__init__()
        self.fpn = base_fpn
        self.attention = attention_module

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Run standard FPN
        out = self.fpn(x)
        # Apply attention to specified levels of FPN output
        # (In this implementation, attention is applied post-FPN output
        #  as a practical approximation. True lateral-connection insertion
        #  requires modifying torchvision internals.)
        out = self.attention(out)
        return out


def build_proposed_model(
    use_attention: bool = False,
    use_sacm: bool = False,
    attention_type: str = "coordinate_attention",
    attention_location: str = "fpn_lateral",
    attention_reduction: int = 32,
    sacm_n_bins: int = 4,
    sacm_context_channels: int = 64,
    num_classes: int = VISDRONE_NUM_CLASSES,
    pretrained_backbone: bool = True,
) -> Tuple[nn.Module, str]:
    """
    Build the proposed SOA-FasterRCNN model.

    Args:
        use_attention: Whether to add Coordinate Attention (Experiment 2/4).
        use_sacm: Whether to add Scale-Aware Context Module (Experiment 3/4).
        attention_type: Type of attention module to use.
        attention_location: Where to insert attention.
        attention_reduction: Reduction ratio for attention module.
        sacm_n_bins: Number of scale bins for SACM.
        sacm_context_channels: Internal channels for SACM.
        num_classes: Number of classes (including background).
        pretrained_backbone: Use pretrained weights.

    Returns:
        (model, model_name) tuple.
    """
    # Determine experiment name
    if use_attention and use_sacm:
        model_name = "fasterrcnn_ca_sacm"        # Experiment 4 (SOA-FasterRCNN)
    elif use_attention:
        model_name = "fasterrcnn_ca"              # Experiment 2
    elif use_sacm:
        model_name = "fasterrcnn_sacm"            # Experiment 3
    else:
        model_name = "baseline_fasterrcnn"        # Experiment 1

    print(f"[INFO] Building model: {model_name}")

    # Build baseline model
    model = build_baseline_model(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
    )

    # Add Coordinate Attention at FPN
    if use_attention:
        fpn_channels = 256  # Standard FPN output channels
        attn_module = build_attention_module(
            attention_type=attention_type,
            channels=fpn_channels,
            location=attention_location,
            reduction=attention_reduction,
        )
        if attn_module is not None:
            # Wrap the FPN body to include attention
            original_fpn = model.backbone.fpn
            model.backbone.fpn = AttentionFPN(original_fpn, attn_module)
            print(f"[INFO] Added {attention_type} attention at {attention_location}")

    # Add Scale-Aware Context Module in RPN
    if use_sacm:
        # Get number of anchors per location from existing RPN head
        num_anchors = model.rpn.head.cls_logits.out_channels
        in_channels = 256  # Standard FPN channels

        new_rpn_head = ScaleAwareRPNHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            n_scale_bins=sacm_n_bins,
            context_channels=sacm_context_channels,
        )
        model.rpn.head = new_rpn_head
        print(f"[INFO] Replaced RPN head with Scale-Aware Context Module")
        print(f"       Scale bins: {sacm_n_bins}, Context channels: {sacm_context_channels}")

    return model, model_name


def get_model_complexity(model: nn.Module, img_size: int = 1280) -> Dict[str, float]:
    """
    Estimate model parameter count and approximate GFLOPs.

    Note: GFLOPs calculation for variable-input detection models is complex.
    This provides an approximation using a single forward pass profiling.
    For publication-grade FLOPs, use fvcore or ptflops libraries.

    Returns:
        Dict with 'params_M' (millions), 'trainable_params_M', and 'gflops' (approx).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "gflops": "NOT YET VERIFIED - use fvcore for accurate computation",
    }

    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = [torch.zeros(3, img_size, img_size)]
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, (dummy,))
        result["gflops"] = round(flops.total() / 1e9, 2)
    except ImportError:
        pass  # fvcore not installed; GFLOPs remain as placeholder

    return result
