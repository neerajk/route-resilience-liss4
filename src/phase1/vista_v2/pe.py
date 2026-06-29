"""Positional encodings for VISTA-v2 — three reusable strategies behind one module.

  sincos_input_channels : INPUT-level absolute PE — sinusoidal coord channels to
      concat to [G,R,NGRDI] (Vaswani 2017). Absolute → cheap but can overfit tile
      layout under spatial-block CV (a documented trade-off; see docs/vista_v2.md).
  RelativePositionBias  : BoTNet/Swin-style learned RELATIVE bias added to attention
      scores (Srinivas 2021; Liu 2021). Translation-robust → best for road continuity.
  RoPE2D                : 2-D rotary PE — rotates q/k by angle ∝ position (Su 2021).
      Relative, param-free, extrapolates to other tile sizes.

Relative PE and RoPE are ATTENTION-INTERNAL (used by attention.py); sinusoidal-input
is the only one that lives at the input.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# sinusoidal INPUT-level PE                                                    #
# --------------------------------------------------------------------------- #
def sincos_input_channels(h: int, w: int, n_freq: int = 2, device=None,
                          dtype=torch.float32) -> torch.Tensor:
    """4*n_freq sinusoidal coord channels -> [1, 4*n_freq, h, w].

    Per axis (x,y) and per frequency: sin and cos of the normalized coordinate.
    Default n_freq=2 -> 8 channels (a small, effective multi-scale default).
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, w).expand(h, w)
    chans = []
    for f in range(n_freq):
        freq = (2.0 ** f) * torch.pi
        for coord in (ys, xs):
            chans.append(torch.sin(freq * coord))
            chans.append(torch.cos(freq * coord))
    return torch.stack(chans, 0).unsqueeze(0)            # [1, 4*n_freq, h, w]


def sincos_n_channels(n_freq: int = 2) -> int:
    return 4 * int(n_freq)


# --------------------------------------------------------------------------- #
# BoTNet/Swin relative position bias (attention-internal)                      #
# --------------------------------------------------------------------------- #
class RelativePositionBias(nn.Module):
    """Learned relative-position bias for an h×w token grid -> [heads, N, N]."""

    def __init__(self, h: int, w: int, num_heads: int) -> None:
        super().__init__()
        self.h, self.w = h, w
        self.table = nn.Parameter(torch.zeros((2 * h - 1) * (2 * w - 1), num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij")).flatten(1)   # [2, N]
        rel = coords[:, :, None] - coords[:, None, :]                       # [2, N, N]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += h - 1
        rel[:, :, 1] += w - 1
        rel[:, :, 0] *= 2 * w - 1
        self.register_buffer("index", rel.sum(-1))                         # [N, N]

    def forward(self) -> torch.Tensor:
        bias = self.table[self.index.view(-1)].view(
            self.h * self.w, self.h * self.w, -1)
        return bias.permute(2, 0, 1).contiguous()                          # [heads, N, N]


# --------------------------------------------------------------------------- #
# 2-D Rotary PE (attention-internal)                                          #
# --------------------------------------------------------------------------- #
class RoPE2D(nn.Module):
    """2-D rotary position embedding applied to q,k. head_dim must be divisible by 4."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("RoPE2D needs head_dim divisible by 4")
        self.head_dim = head_dim
        quarter = head_dim // 4
        inv = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
        self.register_buffer("inv_freq", inv)            # [head_dim/4]

    def _cos_sin(self, h, w, device, dtype):
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        fy = torch.outer(y, self.inv_freq.to(device))    # [h, q]
        fx = torch.outer(x, self.inv_freq.to(device))    # [w, q]
        fy = fy[:, None, :].expand(h, w, -1)
        fx = fx[None, :, :].expand(h, w, -1)
        ang = torch.cat([fy, fx], -1).reshape(h * w, -1) # [N, head_dim/2]
        ang = torch.cat([ang, ang], -1)                  # [N, head_dim]
        return ang.cos().to(dtype), ang.sin().to(dtype)

    @staticmethod
    def _rotate_half(x):
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], -1)

    def forward(self, q, k, h, w):                       # q,k: [B, heads, N, head_dim]
        cos, sin = self._cos_sin(h, w, q.device, q.dtype)
        cos = cos[None, None]; sin = sin[None, None]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k
