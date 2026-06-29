"""Shared backbone building blocks: conv stem, overlapping patch embed, 2-D
sinusoidal positional encoding, and a small MLP — reused by the transformer
backbones (CSWin, HA-RoadFormer).

The 2-D sinusoidal positional encoding is the INPUT-side half of GROVE's sin/cos
idea (the TARGET-side half is the orientation head). It is fixed (not learned),
added to tokens so attention can reason about absolute position when connecting
collinear road pixels across a gap (Vaswani et al. 2017, extended to 2-D).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def sincos_2d(h: int, w: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Fixed 2-D sinusoidal positional encoding -> [1, dim, h, w].

    Splits `dim` in half: first half encodes the y axis, second half the x axis,
    each with the standard sin/cos-of-geometric-frequencies scheme. `dim` must be
    divisible by 4 (two axes × sin/cos).
    """
    if dim % 4 != 0:
        raise ValueError(f"sincos_2d dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=dtype) / max(quarter - 1, 1)
    omega = 1.0 / (10000.0 ** omega)                         # [quarter]
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    sy = torch.einsum("i,j->ij", y, omega)                   # [h, quarter]
    sx = torch.einsum("i,j->ij", x, omega)                   # [w, quarter]
    pe = torch.zeros(dim, h, w, device=device, dtype=dtype)
    pe[0:quarter] = torch.sin(sy).t().unsqueeze(-1).expand(quarter, h, w)
    pe[quarter:2 * quarter] = torch.cos(sy).t().unsqueeze(-1).expand(quarter, h, w)
    pe[2 * quarter:3 * quarter] = torch.sin(sx).t().unsqueeze(-2).expand(quarter, h, w)
    pe[3 * quarter:4 * quarter] = torch.cos(sx).t().unsqueeze(-2).expand(quarter, h, w)
    return pe.unsqueeze(0)                                    # [1, dim, h, w]


class ConvStem(nn.Module):
    """LeViT/HA-RoadFormer-style conv stem: two 3x3 stride-2 convs -> stride-4 map."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = out_ch // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class OverlapPatchEmbed(nn.Module):
    """Overlapping-conv patch embedding (HA-RoadFormer §3.3 / SegFormer).

    A single conv with kernel `k`, stride `s`, padding `k//2` downsamples while
    keeping neighbouring patches overlapping (no hard gridding of thin roads).
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.proj(x))


class MLP(nn.Module):
    """Token MLP (1x1 convs so it operates on [B,C,H,W] feature maps)."""

    def __init__(self, dim: int, hidden: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden = hidden or dim * 4
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))
