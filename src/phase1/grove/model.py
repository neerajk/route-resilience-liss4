"""GroveNet — the assembled GROVE arm: pluggable backbone + shared FPN + heads.

  input [B,in_ch,H,W]
    └─► backbone (vista_mit | vista_resnet | cswin | haroadformer)  → pyramid
          └─► FPN decoder → fused map [B,D,H,W]
                ├─► SegHead         → seg logits [B,1,H,W]
                └─► OrientationHead → [B,2,H,W] (sin2θ,cos2θ)   (if 'orientation' in heads)

forward() returns:
  - a Tensor [B,1,H,W]  when heads == ['seg']  → DROP-IN for the existing VISTA
    trainer/predictor/loss (this is the Stage-2 backbone benchmark path);
  - a dict {'seg':…, 'orient':…}  when the orientation head is enabled  → consumed
    by grove/train.py (the multi-task Stage 3-4 path).

(Optional) CoANet strip-conv + connectivity refinement (Stage 5) is inserted on the
fused decoder map when cfg.grove.coanet.enabled.
"""
from __future__ import annotations

from typing import Any, Dict, Union

import torch
import torch.nn as nn

from .backbones import build_backbone
from .decoder import FPNDecoder
from .heads import OrientationHead, SegHead


class GroveNet(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        g = cfg.get("grove", {}) or {}
        m = cfg.get("model", {}) or {}
        self.heads_cfg = [str(h).lower() for h in g.get("heads", ["seg"])]
        out_dim = int(g.get("decoder_dim", 128))

        self.backbone = build_backbone(cfg)
        red = getattr(self.backbone, "feature_reductions", [4, 8, 16, 32])[0]
        self.decoder = FPNDecoder(self.backbone.feature_channels, out_dim=out_dim,
                                  in_reduction=int(red))

        self.coanet = None
        if (g.get("coanet", {}) or {}).get("enabled", False):
            from .modules.coanet import CoANetRefine
            self.coanet = CoANetRefine(out_dim, **{k: v for k, v in
                                                   (g.get("coanet", {}) or {}).items()
                                                   if k != "enabled"})

        self.seg = SegHead(out_dim, classes=int(m.get("classes", 1)))
        self.orient = OrientationHead(out_dim) if "orientation" in self.heads_cfg else None

    def forward(self, x) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.backbone(x)
        fmap = self.decoder(feats)
        if self.coanet is not None:
            fmap = self.coanet(fmap)
        seg = self.seg(fmap)
        if self.orient is None:                 # seg-only → plain logits (VISTA-compatible)
            return seg
        return {"seg": seg, "orient": self.orient(fmap)}


def build_grove(cfg: Dict[str, Any]) -> nn.Module:
    """Factory entrypoint (called lazily from shared/models/factory.py)."""
    return GroveNet(cfg)
