"""Bottleneck self-attention block for VISTA-v2 (BoTNet-style).

Runs multi-head self-attention over the deepest encoder feature map (stride-32 →
~8×8 = 64 tokens for a 256 tile, so attention is cheap) and injects position via
either a learned RELATIVE bias (`pe='botnet'`) or 2-D RoPE (`pe='rope'`). A
residual + LayerNorm wraps it so it can be dropped onto a pretrained bottleneck
without destabilising it. `pe='none'` = plain attention (unused; sincos/nope skip
this block entirely).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .pe import RelativePositionBias, RoPE2D


class BottleneckAttention(nn.Module):
    def __init__(self, dim: int, feat_h: int, feat_w: int, num_heads: int = 8,
                 pe: str = "botnet") -> None:
        super().__init__()
        self.h, self.w, self.heads = feat_h, feat_w, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pe = pe
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel = RelativePositionBias(feat_h, feat_w, num_heads) if pe == "botnet" else None
        self.rope = RoPE2D(self.head_dim) if pe == "rope" else None

    def forward(self, x):                                 # x: [B, C, H, W]
        B, C, H, W = x.shape
        t = rearrange(x, "b c h w -> b (h w) c")
        t = self.norm(t)
        qkv = self.qkv(t)
        q, k, v = rearrange(qkv, "b n (three heads d) -> three b heads n d",
                            three=3, heads=self.heads)
        if self.rope is not None:
            q, k = self.rope(q, k, H, W)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel is not None:
            attn = attn + self.rel().unsqueeze(0)        # [1, heads, N, N]
        attn = attn.softmax(-1)
        out = rearrange(attn @ v, "b heads n d -> b n (heads d)")
        out = self.proj(out)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return x + out                                   # residual
