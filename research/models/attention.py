"""
Attention Modules for Small-Object Detection in UAV Imagery.

This module implements candidate attention mechanisms investigated for the
research project. Each module is documented with:
  - Original paper reference
  - Classification: EXISTING METHOD or POTENTIAL NOVELTY
  - Suitability analysis for UAV small-object detection
  - Selected insertion point justification

Literature Review Summary:
--------------------------
CBAM (Woo et al., ECCV 2018): EXISTING METHOD
  - Sequential channel + spatial attention
  - Widely used in UAV detection (various YOLO variants, custom detectors)
  - Limitation for small objects: global average pool in channel branch loses
    fine-grained spatial structure critical for sub-16px objects

SE Net (Hu et al., CVPR 2018): EXISTING METHOD
  - Channel attention only
  - No spatial information preserved
  - Not ideal for localization-sensitive small object tasks

ECA Net (Wang et al., CVPR 2020): EXISTING METHOD
  - Efficient channel attention (1D conv, no dimensionality reduction)
  - More parameter-efficient than SE/CBAM
  - Still lacks spatial component needed for precise localization

Coordinate Attention (Hou et al., CVPR 2021): EXISTING METHOD
  - Decomposes 2D feature map into two 1D pools (horizontal + vertical)
  - Preserves positional information better than CBAM/SE
  - Shown effective for mobile detection tasks
  - POTENTIAL ADAPTATION: Using CA within FPN lateral connections
    specifically for P2/P3 levels in Faster R-CNN context on VisDrone
    requires experimental validation — POTENTIAL NOVELTY — REQUIRES VERIFICATION

Selected Approach:
------------------
Coordinate Attention at FPN lateral connections (P2 and P3 levels).

Justification:
  1. FPN lateral connections at P2/P3 carry the highest spatial resolution
     features, which are most informative for small objects (<32px).
  2. Standard FPN merges these features with top-down semantic features
     without any recalibration — CA can selectively emphasize object-relevant
     channels before this fusion step.
  3. Coordinate Attention's positional encoding component helps preserve
     spatial precision required for tight bounding boxes on tiny objects.
  4. Lower computational overhead than CBAM for the same improvement
     (no spatial map computation, only 1D pooling).

Note: This placement is DIFFERENT from applying attention at FPN output
(after all fusion), which is the more common approach in existing literature.
The lateral-connection-specific adaptation is the proposed contribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================
# 1. SE Module (Hu et al., 2018) — EXISTING METHOD
# ============================================================
class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Network channel attention.
    Reference: Hu et al., 'Squeeze-and-Excitation Networks', CVPR 2018.
    Classification: EXISTING METHOD
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ============================================================
# 2. CBAM (Woo et al., 2018) — EXISTING METHOD
# ============================================================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Reference: Woo et al., 'CBAM: Convolutional Block Attention Module', ECCV 2018.
    Classification: EXISTING METHOD

    Limitation for small UAV objects: global average+max pool in channel branch
    loses positional information. Spatial attention map is computed from a 7x7
    kernel which may be too coarse for very small objects at P2 resolution.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.channel_mlp_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Channel attention
        avg_out = self.channel_mlp(x)
        max_out = self.channel_mlp_max(x)
        ch_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * ch_att
        # Spatial attention
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        sp_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * sp_att


# ============================================================
# 3. ECA Module (Wang et al., 2020) — EXISTING METHOD
# ============================================================
class ECAModule(nn.Module):
    """
    Efficient Channel Attention.
    Reference: Wang et al., 'ECA-Net: Efficient Channel Attention', CVPR 2020.
    Classification: EXISTING METHOD
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        import math
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * torch.sigmoid(y)


# ============================================================
# 4. Coordinate Attention (Hou et al., 2021) — EXISTING METHOD
#    Our adaptation for FPN lateral connections: POTENTIAL NOVELTY
# ============================================================
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention module.
    Reference: Hou et al., 'Coordinate Attention for Efficient Mobile Network Design',
               CVPR 2021. Classification: EXISTING METHOD.

    Our adaptation:
    - Applied at FPN lateral connection outputs (before top-down fusion)
    - Specifically at P2/P3 levels for small-object calibration
    - This specific combination (CA + FPN lateral + Faster RCNN + VisDrone sub-32px)
      requires experimental validation.
    - POTENTIAL NOVELTY — REQUIRES VERIFICATION

    Why Coordinate Attention over CBAM for this task:
    1. CA preserves spatial coordinate information (x,y position) which is critical
       for precise localization of tiny objects.
    2. CA uses 1D pooling (H-direction and W-direction separately) rather than 2D
       global pooling, preserving positional cues lost in CBAM's global pool.
    3. Lower parameter overhead than CBAM for the same channel count.
    """

    def __init__(self, in_channels: int, reduction: int = 32) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape

        x_h = self.pool_h(x)          # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]

        # Concatenate along spatial dimension, process jointly
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [B, mid, H+W, 1]

        x_h_att, x_w_att = torch.split(y, [h, w], dim=2)
        x_w_att = x_w_att.permute(0, 1, 3, 2)

        att_h = torch.sigmoid(self.conv_h(x_h_att))  # [B, C, H, 1]
        att_w = torch.sigmoid(self.conv_w(x_w_att))  # [B, C, 1, W]

        return identity * att_h * att_w


# ============================================================
# 5. FPN Lateral Attention Wrapper
# ============================================================
class FPNLateralAttention(nn.Module):
    """
    Wraps any attention module to be inserted at FPN lateral connections.

    This wrapper applies attention to specified FPN pyramid levels.
    For small-object detection, we target P2 and P3 (highest resolution levels).

    Design rationale:
    - Lateral connections in FPN fuse bottom-up (high-resolution) features
      with top-down (high-semantic) features.
    - Applying attention BEFORE this fusion allows the module to select
      which spatial positions and channels in the high-resolution features
      are most informative for small objects.
    - This differs from applying attention at the FPN OUTPUT (post-fusion),
      which is the standard approach in most literature.

    Args:
        channels: Number of feature channels (256 for standard FPN).
        attention_type: Which attention module to use.
        target_levels: Which FPN levels to apply attention to.
                       Defaults to [0, 1] = P2, P3 (small object levels).
        reduction: Reduction ratio for the attention module.
    """

    def __init__(
        self,
        channels: int = 256,
        attention_type: str = "coordinate_attention",
        target_levels: Optional[list] = None,
        reduction: int = 32,
    ) -> None:
        super().__init__()

        if target_levels is None:
            target_levels = [0, 1]  # P2 and P3 by default
        self.target_levels = target_levels

        attention_map = {
            "coordinate_attention": lambda: CoordinateAttention(channels, reduction),
            "cbam": lambda: CBAM(channels, reduction),
            "se": lambda: SEModule(channels, reduction),
            "eca": lambda: ECAModule(channels),
            "none": lambda: nn.Identity(),
        }
        if attention_type not in attention_map:
            raise ValueError(
                f"Unknown attention type '{attention_type}'. "
                f"Choose from: {list(attention_map.keys())}"
            )

        # Create one attention module per target level
        self.attention_modules = nn.ModuleList([
            attention_map[attention_type]() for _ in target_levels
        ])

        self.attention_type = attention_type

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply attention to specified FPN levels.

        Args:
            features: OrderedDict from FPN {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}

        Returns:
            Modified features dict with attention applied at target levels.
        """
        feature_keys = [k for k in features.keys() if k != "pool"]
        out = dict(features)

        for level_idx, (target, attn_module) in enumerate(
            zip(self.target_levels, self.attention_modules)
        ):
            if target < len(feature_keys):
                key = feature_keys[target]
                out[key] = attn_module(features[key])

        return out


# ============================================================
# Factory function
# ============================================================
def build_attention_module(
    attention_type: str,
    channels: int = 256,
    location: str = "fpn_lateral",
    reduction: int = 32,
) -> Optional[nn.Module]:
    """
    Build the appropriate attention module based on config.

    Args:
        attention_type: "coordinate_attention", "cbam", "se", "eca", or "none"
        channels: Feature map channels (256 for standard FPN)
        location: Where attention is inserted ("fpn_lateral" is our proposed location)
        reduction: Attention reduction ratio

    Returns:
        Attention module or None if attention_type is "none"
    """
    if attention_type == "none":
        return None

    if location == "fpn_lateral":
        return FPNLateralAttention(
            channels=channels,
            attention_type=attention_type,
            target_levels=[0, 1],  # P2, P3
            reduction=reduction,
        )
    elif location == "fpn_output":
        return FPNLateralAttention(
            channels=channels,
            attention_type=attention_type,
            target_levels=[0, 1, 2, 3],  # All output levels
            reduction=reduction,
        )
    else:
        raise ValueError(f"Unsupported location: {location}. Use 'fpn_lateral' or 'fpn_output'.")
