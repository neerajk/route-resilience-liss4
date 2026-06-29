"""Backbone-agnostic GROVE decoder — a small top-down FPN.

Consumes any backbone's feature pyramid (fine→coarse list of [B,C_i,H/s_i,W/s_i])
and returns a single fused feature map at FULL input resolution, on which the seg
and orientation heads sit. Backbone-agnostic by construction (it only needs the
per-level channel counts), which is what lets VISTA/CSWin/HA-RoadFormer share it.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class FPNDecoder(nn.Module):
    """Top-down FPN → fused map upsampled to full input resolution.

    Parameters
    ----------
    feature_channels : per-level channels of the backbone pyramid (fine→coarse).
    out_dim : decoder width D (head input channels).
    in_reduction : stride of the FINEST pyramid level (4 for these backbones) —
        the final upsample factor back to full resolution.
    """

    def __init__(self, feature_channels: List[int], out_dim: int = 128,
                 in_reduction: int = 4) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_dim, 1) for c in feature_channels])
        self.smooth = nn.ModuleList([_ConvBNAct(out_dim, out_dim) for _ in feature_channels])
        self.in_reduction = int(in_reduction)
        self.head_proj = nn.Sequential(_ConvBNAct(out_dim, out_dim), _ConvBNAct(out_dim, out_dim))

    def forward(self, feats):                       # feats: fine→coarse
        laterals = [l(f) for l, f in zip(self.laterals, feats)]
        p = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):  # coarse→fine top-down add
            p = F.interpolate(p, size=laterals[i].shape[-2:], mode="bilinear",
                              align_corners=False)
            p = self.smooth[i](laterals[i] + p)
        # p is at the finest pyramid stride (in_reduction); lift to full resolution
        if self.in_reduction > 1:
            p = F.interpolate(p, scale_factor=self.in_reduction, mode="bilinear",
                              align_corners=False)
        return self.head_proj(p)
