"""CSWin backbone — Cross-Shaped Window attention (Dong et al., CVPR 2022).

Cross-shaped window self-attention computes attention in HORIZONTAL stripes
(sw rows × full width) and VERTICAL stripes (full height × sw cols) IN PARALLEL on
two halves of the heads, then concatenates. The cross shape matches road geometry
(long horizontal/vertical extents), which is why it's a natural road backbone
(DCS-TransUperNet, Zhang et al. 2022).

Compact reimplementation: LePE (locally-enhanced positional encoding) is replaced
by the input-side 2-D sinusoidal PE added in the backbone (base.sincos_2d). Public
CSWin ImageNet weights would need the official checkpoint mapping — `weights_path`
is a load hook; default is from-scratch (flagged for the benchmark).

Output: feature pyramid (fine→coarse) at strides [4, 8, 16, 32].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import MLP, ConvStem, OverlapPatchEmbed, sincos_2d
from .haroadformer import _mhsa


def _stripe_attention(x, qkv_lin, stripe_h, stripe_w, num_heads):
    """Window/stripe attention over rectangular (stripe_h × stripe_w) windows.

    x: [B,C,H,W]; qkv_lin: Linear(C, 3C). Pads to a multiple of the stripe.
    """
    B, C, H, W = x.shape
    sh = min(stripe_h, H) if stripe_h > 0 else H
    sw = min(stripe_w, W) if stripe_w > 0 else W
    ph, pw = (sh - H % sh) % sh, (sw - W % sw) % sw
    xp = F.pad(x, (0, pw, 0, ph))
    Hp, Wp = xp.shape[-2:]
    win = rearrange(xp, "b c (nh sh) (nw sw) -> (b nh nw) (sh sw) c", sh=sh, sw=sw)
    q, k, v = qkv_lin(win).chunk(3, dim=-1)
    out = _mhsa(q, k, v, num_heads)
    out = rearrange(out, "(b nh nw) (sh sw) c -> b c (nh sh) (nw sw)",
                    b=B, nh=Hp // sh, nw=Wp // sw, sh=sh, sw=sw)
    return out[:, :, :H, :W]


class CSWinAttention(nn.Module):
    """Half heads → horizontal stripes, half → vertical stripes, concatenated."""

    def __init__(self, dim: int, num_heads: int = 4, stripe: int = 7) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("CSWin dim must be even (split into H/V halves)")
        self.half = dim // 2
        self.heads = max(1, num_heads // 2)
        self.stripe = stripe
        self.qkv_h = nn.Linear(self.half, self.half * 3)   # horizontal branch
        self.qkv_v = nn.Linear(self.half, self.half * 3)   # vertical branch
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                                  # [B,C,H,W]
        xh, xv = x[:, :self.half], x[:, self.half:]
        oh = _stripe_attention(xh, self.qkv_h, self.stripe, 0, self.heads)  # sw rows × full W
        ov = _stripe_attention(xv, self.qkv_v, 0, self.stripe, self.heads)  # full H × sw cols
        out = torch.cat([oh, ov], dim=1)
        out = rearrange(out, "b c h w -> b (h w) c")
        out = self.proj(out)
        H, W = x.shape[-2:]
        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)


class CSWinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, stripe=7) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = CSWinAttention(dim, num_heads, stripe)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CSWinBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, dims=(64, 128, 256, 512),
                 depths=(2, 2, 4, 2), num_heads=(2, 4, 8, 16), stripes=(1, 2, 7, 7),
                 use_pos: bool = True, weights_path: str | None = None) -> None:
        super().__init__()
        self.use_pos = use_pos
        self.stem = ConvStem(in_channels, dims[0])             # stride 4
        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = dims[0]
        for i, d in enumerate(dims):
            self.embeds.append(nn.Identity() if i == 0
                               else OverlapPatchEmbed(prev, d, k=3, s=2))
            self.stages.append(nn.ModuleList(
                [CSWinBlock(d, num_heads[i], stripes[i]) for _ in range(depths[i])]))
            prev = d
        self.feature_channels = list(dims)
        self.feature_reductions = [4, 8, 16, 32]
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(sd.get("model", sd), strict=False)

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
        return feats
