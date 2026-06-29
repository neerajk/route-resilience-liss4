"""HA-RoadFormer backbone (Zhang, Miao, Liu, Tian, Zhou — Mathematics 2022, 10:1915).

Faithful-but-compact reimplementation of the paper's three signatures:
  1. CONV STEM (two 3x3 stride-2) -> stride-4 feature map (LeViT-style).
  2. MULTI-SCALE OVERLAPPING PATCH EMBED — parallel overlapping convs with kernels
     {3,5,7} at the SAME stride/padding so they yield the SAME sequence length, then
     summed (the paper's coarse-to-fine multi-branch patch embedding, §3.3).
  3. HYBRID ATTENTION (linear-ish) = ADJACENT (local window) + LONG-RANGE attention.
     The paper uses banded-diagonal (adjacent) + RANDOM keys; we realise the same
     intent — strong local inductive bias + long-distance dependence — with window
     attention + pooled global tokens (deterministic, dependency-light). Documented
     deviation: pooled-global stands in for the paper's random-key sampling.

NOTE ON WEIGHTS: there is no public HA-RoadFormer checkpoint, so this trains from
scratch (or via DeepGlobe pretrain). That is itself a benchmark finding vs the
ImageNet-pretrained backbones — report it, don't hide it.

Output: feature pyramid (fine→coarse) at strides [4, 8, 16, 32].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import MLP, ConvStem, OverlapPatchEmbed, sincos_2d


class MultiScalePatchEmbed(nn.Module):
    """Parallel overlapping-conv patch embeds (k=3,5,7), same output size, summed."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2,
                 kernels=(3, 5, 7)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [OverlapPatchEmbed(in_ch, out_ch, k=k, s=stride) for k in kernels])

    def forward(self, x):
        out = 0
        for b in self.branches:
            out = out + b(x)
        return out / len(self.branches)


def _mhsa(q, k, v, num_heads: int):
    """Multi-head scaled-dot-product attention on [B,N,C] tensors -> [B,N,C]."""
    b, nq, c = q.shape
    h = num_heads
    q = rearrange(q, "b n (h d) -> b h n d", h=h)
    k = rearrange(k, "b n (h d) -> b h n d", h=h)
    v = rearrange(v, "b n (h d) -> b h n d", h=h)
    attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    attn = attn.softmax(dim=-1)
    out = attn @ v
    return rearrange(out, "b h n d -> b n (h d)")


class HybridAttention(nn.Module):
    """Adjacent (local window) + long-range (pooled global) attention. Linear-ish."""

    def __init__(self, dim: int, num_heads: int = 4, window: int = 7,
                 global_tokens: int = 8) -> None:
        super().__init__()
        self.dim, self.num_heads, self.window = dim, num_heads, window
        self.g = int(global_tokens)
        self.qkv_local = nn.Linear(dim, dim * 3)
        self.q_glob = nn.Linear(dim, dim)
        self.kv_glob = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def _window(self, x):                       # x:[B,C,H,W] -> local-attended [B,C,H,W]
        B, C, H, W = x.shape
        ws = self.window
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        xp = F.pad(x, (0, pw, 0, ph))
        Hp, Wp = xp.shape[-2:]
        win = rearrange(xp, "b c (nh ws1) (nw ws2) -> (b nh nw) (ws1 ws2) c",
                        ws1=ws, ws2=ws)
        qkv = self.qkv_local(win).chunk(3, dim=-1)
        out = _mhsa(qkv[0], qkv[1], qkv[2], self.num_heads)
        out = rearrange(out, "(b nh nw) (ws1 ws2) c -> b c (nh ws1) (nw ws2)",
                        b=B, nh=Hp // ws, nw=Wp // ws, ws1=ws, ws2=ws)
        return out[:, :, :H, :W]

    def _global(self, x):                       # long-range via pooled key tokens
        B, C, H, W = x.shape
        g = self.g
        kv_src = F.adaptive_avg_pool2d(x, (g, g))               # [B,C,g,g]
        kv_src = rearrange(kv_src, "b c gh gw -> b (gh gw) c")
        q = rearrange(x, "b c h w -> b (h w) c")
        qg = self.q_glob(q)
        kg, vg = self.kv_glob(kv_src).chunk(2, dim=-1)
        out = _mhsa(qg, kg, vg, self.num_heads)
        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

    def forward(self, x):                       # x:[B,C,H,W]
        local = self._window(x)
        glob = self._global(x)
        fused = rearrange(local + glob, "b c h w -> b (h w) c")
        fused = self.proj(fused)
        H, W = x.shape[-2:]
        return rearrange(fused, "b (h w) c -> b c h w", h=H, w=W)


class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window=7, global_tokens=8) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = HybridAttention(dim, num_heads, window, global_tokens)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HARoadFormerBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, dims=(64, 128, 256, 512),
                 depths=(2, 2, 4, 2), num_heads=(2, 4, 8, 8), window: int = 7,
                 global_tokens: int = 8, use_pos: bool = True) -> None:
        super().__init__()
        self.use_pos = use_pos
        self.stem = ConvStem(in_channels, dims[0])              # stride 4
        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = dims[0]
        for i, d in enumerate(dims):
            # stage 0 keeps stride 4 (no downsample); stages 1..3 halve via patch embed
            self.embeds.append(nn.Identity() if i == 0
                               else MultiScalePatchEmbed(prev, d, stride=2))
            self.stages.append(nn.ModuleList(
                [HybridBlock(d, num_heads[i], window, global_tokens) for _ in range(depths[i])]))
            prev = d
        self.feature_channels = list(dims)
        self.feature_reductions = [4, 8, 16, 32]

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for embed, blocks in zip(self.embeds, self.stages):
            x = embed(x)
            if self.use_pos:
                x = x + sincos_2d(x.shape[-2], x.shape[-1], x.shape[1],
                                  device=x.device, dtype=x.dtype)
            for blk in blocks:
                x = blk(x)
            feats.append(x)
        return feats                                            # fine→coarse
