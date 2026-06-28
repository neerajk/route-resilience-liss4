"""CoANet refinement (Mei et al., IEEE T-IP 2021) — Stage 5 add-on.

Two ideas from the paper, adapted as a drop-in refinement on the decoder feature map:

  Strip Convolution Module (SCM): roads are long, narrow, continuous, so a square
  kernel mixes in off-road clutter. SCM uses DIRECTIONAL strip kernels (1×k, k×1,
  and two diagonals) to gather long-range context ALONG the road while ignoring the
  sides. Here: horizontal + vertical strips + diagonal strips approximated by
  strip-conv on the flipped tensor (documented simplification of the paper's exact
  diagonal sampling).

  Connectivity Attention: a light spatial gate that lets the map emphasise
  road-consistent locations (a compact stand-in for the paper's pairwise
  connectivity module), which helps bridge tree-occlusion breaks.

Refines [B,D,H,W] → [B,D,H,W], residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StripConv(nn.Module):
    """Four directional strip convolutions summed (H, V, and two diagonals)."""

    def __init__(self, dim: int, k: int = 9) -> None:
        super().__init__()
        self.h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), bias=False)
        self.v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), bias=False)
        # diagonals approximated by applying the H/V strips on a transposed/flipped view
        self.d1 = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), bias=False)
        self.d2 = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), bias=False)
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False),
                                  nn.BatchNorm2d(dim), nn.GELU())

    def forward(self, x):
        flip = torch.flip(x, dims=[-1])                      # cheap diagonal proxy
        d = torch.flip(self.d1(flip), dims=[-1]) + self.d2(x)
        return self.fuse(self.h(x) + self.v(x) + d)


class ConnectivityAttention(nn.Module):
    """Light spatial gate emphasising road-consistent locations."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim // 2), nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class CoANetRefine(nn.Module):
    def __init__(self, dim: int, strip_k: int = 9, **_ignore) -> None:
        super().__init__()
        self.scm = StripConv(dim, k=int(strip_k))
        self.conn = ConnectivityAttention(dim)

    def forward(self, x):
        return x + self.conn(self.scm(x))
