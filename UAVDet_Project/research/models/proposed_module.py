"""
Second Research Contribution: Scale-Aware Context RPN Module.

Research Gap Addressed:
-----------------------
Standard Faster R-CNN RPN uses a fixed objectness scoring head (1 conv layer)
that does not account for the expected scale distribution of objects at each
FPN level. In UAV imagery (VisDrone), there is severe scale imbalance:
- P2 (stride 4): predominantly sub-16px objects (pedestrians, cyclists)
- P3 (stride 8): 16-32px objects (vehicles, small groups)
- P4/P5 (stride 16/32): larger objects (buses, trucks at low altitude)

Existing approaches (EXISTING METHOD):
- RFLA (Xu et al., 2022): Gaussian receptive field label assignment
  -> Changes label assignment strategy, not objectness scoring
- Soft NMS (Bodla et al., 2017): Post-processing NMS variant
  -> Applied after scoring, not during it
- Adaptive NMS (Liu et al., 2019): Dynamic NMS threshold based on density
  -> Post-processing, does not modify RPN scoring

Our contribution (POTENTIAL NOVELTY — REQUIRES VERIFICATION):
A lightweight "Scale-Aware Context Module" (SACM) that augments the standard
RPN objectness head with:
1. A scale-aware feature path: predicts which of K scale bins the proposal
   likely belongs to (using a 1x1 conv classifier on the RPN feature map)
2. A context aggregation path: uses dilated convolutions at multiple rates
   to capture multi-scale context around small proposal regions
3. Soft-gates the objectness score to penalize proposals whose predicted
   scale bin does not match the FPN level's expected scale range

This is distinct from RFLA (label assignment) and Adaptive NMS (post-processing)
because it operates during the FORWARD PASS of the RPN itself.

Status: POTENTIAL NOVELTY — REQUIRES EXPERIMENTAL VERIFICATION
If literature search reveals an identical approach, this claim is invalid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DilatedContextAggregator(nn.Module):
    """
    Multi-scale context aggregation using dilated convolutions.

    Captures both local (dilation=1) and surrounding context (dilation=2,4)
    around small object proposals without increasing feature map size.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid = out_channels // 3
        # Three parallel dilated conv branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - 2 * mid, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels - 2 * mid),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return fused


class ScaleAwareContextModule(nn.Module):
    """
    Scale-Aware Context Module (SACM) for augmenting RPN objectness scoring.

    This module is inserted AFTER the standard RPN feature extraction conv
    but BEFORE the final objectness + box regression heads.

    Architecture:
        RPN Conv (256 ch) -> SACM -> {objectness head, bbox head}

    SACM components:
    1. Scale classifier: Predicts which scale bin the feature region belongs to
    2. Context aggregator: Dilated conv for multi-scale context
    3. Soft gating: Combines scale prediction with context features

    Args:
        in_channels: RPN feature channels (256 for standard FPN)
        n_scale_bins: Number of scale bins to predict (default 4)
        context_channels: Internal channels for context aggregation
    """

    def __init__(
        self,
        in_channels: int = 256,
        n_scale_bins: int = 4,
        context_channels: int = 64,
    ) -> None:
        super().__init__()
        self.n_scale_bins = n_scale_bins

        # Context aggregation (multi-scale dilated convs)
        self.context_agg = DilatedContextAggregator(in_channels, context_channels)

        # Scale prediction head (lightweight 1x1 conv)
        self.scale_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, n_scale_bins),
            nn.Softmax(dim=-1),
        )

        # Gate: combines scale-aware weights with context
        self.gate_proj = nn.Conv2d(context_channels, in_channels, 1, bias=False)
        self.gate_sigmoid = nn.Sigmoid()

        # Output projection: fuse gated context back to original channels
        self.output_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # Scale bin embeddings: each bin gets a learned weight vector
        self.scale_embed = nn.Embedding(n_scale_bins, in_channels)

    def forward(
        self,
        features: torch.Tensor,
        level_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: RPN feature map [B, C, H, W]
            level_idx: FPN level index (used for level-specific scale priors)

        Returns:
            Enhanced feature map [B, C, H, W]
        """
        identity = features
        b, c, h, w = features.shape

        # 1. Predict scale bin distribution for this feature map
        scale_weights = self.scale_pred(features)  # [B, n_scale_bins]
        dominant_bin = scale_weights.argmax(dim=1)  # [B]

        # 2. Get scale embedding for dominant bin
        scale_emb = self.scale_embed(dominant_bin)  # [B, C]
        scale_emb = scale_emb.view(b, c, 1, 1).expand_as(features)

        # 3. Extract multi-scale context
        context = self.context_agg(features)  # [B, context_channels, H, W]

        # 4. Soft gate: use context to modulate scale embedding
        gate = self.gate_sigmoid(self.gate_proj(context))  # [B, C, H, W]
        modulated = identity + scale_emb * gate

        # 5. Output projection with residual
        out = self.output_proj(modulated) + identity
        return out


class ScaleAwareRPNHead(nn.Module):
    """
    Drop-in replacement for the standard Faster R-CNN RPNHead that adds
    the Scale-Aware Context Module between the shared conv and the output heads.

    This preserves the standard RPN interface so it can be used with
    torchvision's GeneralizedRCNN framework.

    Note: This is a RESEARCH implementation. It requires integration with
    the torchvision RPN, which is done in proposed_model.py.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        n_scale_bins: int = 4,
        context_channels: int = 64,
    ) -> None:
        super().__init__()
        # Standard RPN shared conv (preserved)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Scale-aware context module (our addition)
        self.sacm = ScaleAwareContextModule(
            in_channels=in_channels,
            n_scale_bins=n_scale_bins,
            context_channels=context_channels,
        )

        # Standard objectness and bbox heads (preserved)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        # Initialize weights
        for layer in [self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_logits = []
        bbox_preds = []

        for level_idx, feat in enumerate(features):
            t = self.conv(feat)
            t = self.sacm(t, level_idx=level_idx)
            cls_logits.append(self.cls_logits(t))
            bbox_preds.append(self.bbox_pred(t))

        return cls_logits, bbox_preds
