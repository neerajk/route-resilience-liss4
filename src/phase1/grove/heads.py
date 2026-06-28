"""GROVE prediction heads — segmentation and orientation.

  SegHead         : 1x1 conv -> road logit [B,1,H,W]  (sigmoid at loss/inference)
  OrientationHead : 1x1 conv -> [B,2,H,W] = (sin2θ, cos2θ), the axial road-direction
                    field. Raw (un-normalised) output; the loss L2-normalises it to a
                    unit vector and compares (cosine) only on road pixels.
"""
from __future__ import annotations

import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, dim: int, classes: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(dim, classes, 1)

    def forward(self, x):
        return self.head(x)


class OrientationHead(nn.Module):
    """Predicts the 2-channel (sin2θ, cos2θ) axial orientation field."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(dim, 2, 1)

    def forward(self, x):
        return self.head(x)
