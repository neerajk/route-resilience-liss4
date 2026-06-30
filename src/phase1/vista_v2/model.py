"""VistaV2Net — ResNet-101 + UNet++ (smp) with a pluggable positional encoding.

Reuses smp's UnetPlusPlus end-to-end (encoder · decoder · head) and only intercepts
the deepest feature to insert an attention bottleneck (botnet/rope), or concatenates
sinusoidal PE channels at the input (sincos). Output = road logits [B,1,H,W], so the
existing VISTA train/predict/pretrain pipeline drives every variant unchanged.

cfg.model.pe.type: botnet | rope | sincos | nope     (default botnet)
cfg.model.pe.heads: bottleneck attention heads (botnet/rope), default 8
cfg.model.pe.sincos_freqs: frequencies for input PE (sincos), default 2 -> 8 channels
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .attention import BottleneckAttention
from .pe import sincos_input_channels, sincos_n_channels


class VistaV2Net(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        m = cfg.get("model", {}) or {}
        pe = (m.get("pe", {}) or {})
        self.pe_type = str(pe.get("type", "botnet")).lower()
        self.sincos_freqs = int(pe.get("sincos_freqs", 2))
        base_in = int(m.get("in_channels", 3))
        classes = int(m.get("classes", 1))
        tile = int(cfg.get("data", {}).get("tile_size", 256))
        feat = max(1, tile // 32)                         # stride-32 bottleneck grid

        eff_in = base_in + (sincos_n_channels(self.sincos_freqs)
                            if self.pe_type == "sincos" else 0)
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:  # pragma: no cover
            raise ImportError("arch='vista_v2' needs segmentation_models_pytorch.") from e
        self.net = smp.UnetPlusPlus(
            encoder_name=str(m.get("encoder", "resnet101")),
            encoder_weights=m.get("encoder_weights", "imagenet"),
            in_channels=eff_in, classes=classes)

        self.attn = None
        if self.pe_type in ("botnet", "rope"):
            dim = self.net.encoder.out_channels[-1]       # 2048 for resnet101
            self.attn = BottleneckAttention(dim, feat, feat,
                                            num_heads=int(pe.get("heads", 8)), pe=self.pe_type)

    def forward(self, x):
        if self.pe_type == "sincos":
            pe = sincos_input_channels(x.shape[-2], x.shape[-1], self.sincos_freqs,
                                       device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, pe], dim=1)
        feats = self.net.encoder(x)
        if self.attn is not None:
            feats = list(feats)
            feats[-1] = self.attn(feats[-1])
        dec = self.net.decoder(feats)   # smp>=0.4 takes the feature LIST as one arg (not *unpacked)
        return self.net.segmentation_head(dec)


def build_vista_v2(cfg: Dict[str, Any]) -> nn.Module:
    return VistaV2Net(cfg)
