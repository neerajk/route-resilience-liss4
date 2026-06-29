"""GROVE backbones — interchangeable encoders behind one feature-pyramid contract.

Every backbone maps input `[B, in_ch, H, W]` to a list of feature maps
(coarse→fine or fine→coarse, documented per backbone) and exposes:
    .feature_channels : list[int]   channel count of each returned map
    .feature_reductions : list[int] stride (H/feat_H) of each returned map
so the shared GROVE decoder (src/phase1/grove/decoder.py) is backbone-agnostic.
This is what makes the VISTA-vs-CSWin-vs-HA-RoadFormer benchmark fair: only the
backbone changes; heads, decoder, loss, supervision are identical.

Registry (cfg.grove.backbone):
    vista_mit | vista_resnet : smp encoder (MiT-B2 / ResNet34) — ImageNet weights free
    cswin                    : Cross-Shaped Window transformer (Dong et al. 2022)
    haroadformer             : Hybrid-Attention multi-branch (Zhang et al. 2022)
"""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn


def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the backbone named by cfg.grove.backbone. Lazy per-backbone imports."""
    g = cfg.get("grove", {}) or {}
    name = str(g.get("backbone", "vista_mit")).lower()
    in_ch = int(cfg.get("model", {}).get("in_channels", 4))

    if name in ("vista_mit", "vista_resnet", "vista"):
        from .vista_encoder import VistaEncoderBackbone
        encoder = {"vista_mit": "mit_b2", "vista": "mit_b2",
                   "vista_resnet": "resnet34"}[name]
        return VistaEncoderBackbone(
            in_channels=in_ch, encoder_name=str(g.get("encoder", encoder)),
            weights=g.get("encoder_weights", "imagenet"))

    if name == "cswin":
        from .cswin import CSWinBackbone
        return CSWinBackbone(in_channels=in_ch, **(g.get("cswin", {}) or {}))

    if name in ("haroadformer", "ha_roadformer", "haroad"):
        from .haroadformer import HARoadFormerBackbone
        return HARoadFormerBackbone(in_channels=in_ch, **(g.get("haroadformer", {}) or {}))

    raise ValueError(f"Unknown grove.backbone='{name}'. "
                     "Options: vista_mit | vista_resnet | cswin | haroadformer.")
