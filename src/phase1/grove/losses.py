"""GROVE multi-task loss (Stage 4).

  L = w_bce·BCE* + w_dice·Dice + w_cldice·clDice            (segmentation)
      + λ_orient·L_orient                                   (orientation head)

BCE* is per-pixel reweighted to push Occlusion-Recall:
    weight = 1 + canopy_weight·canopy + ucr_weight·under_canopy_road
so missing a road that is BOTH road AND under canopy (the GROVE target) is
penalised hardest. Dice + clDice (Shit et al. 2021) are reused unchanged from the
shared losses, so GROVE's topology term is identical to VISTA's (fair ablation).

L_orient: on road pixels only, 1 − cosine(unit(pred), target) for the axial
(sin2θ,cos2θ) field (Batra et al. 2019). Off-road pixels are ignored (orientation
is undefined there); the prediction is L2-normalised to a unit vector first.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.losses.losses import DiceLoss, SoftClDiceLoss


class GroveLoss(nn.Module):
    def __init__(self, w_bce: float = 0.3, w_dice: float = 0.4, w_cldice: float = 0.3,
                 cldice_iters: int = 6, canopy_weight: float = 1.5,
                 ucr_weight: float = 2.0, orient_weight: float = 0.5) -> None:
        super().__init__()
        self.w_bce, self.w_dice, self.w_cldice = w_bce, w_dice, w_cldice
        self.canopy_weight = float(canopy_weight)
        self.ucr_weight = float(ucr_weight)
        self.orient_weight = float(orient_weight)
        self.dice = DiceLoss()
        self.cldice = SoftClDiceLoss(iters=cldice_iters)

    def _orient_loss(self, pred, target, mask, eps: float = 1e-6):
        """1 − cosine on road pixels for the (sin2θ,cos2θ) field."""
        pred_u = pred / (pred.norm(dim=1, keepdim=True) + eps)      # unit vector
        cos = (pred_u * target).sum(dim=1, keepdim=True)           # [B,1,H,W]
        m = (mask >= 0.5).float()
        denom = m.sum().clamp_min(1.0)
        return ((1.0 - cos) * m).sum() / denom

    def forward(self, out: Union[torch.Tensor, Dict[str, torch.Tensor]],
                mask: torch.Tensor, canopy: Optional[torch.Tensor] = None,
                under_canopy: Optional[torch.Tensor] = None,
                orient_target: Optional[torch.Tensor] = None):
        seg = out["seg"] if isinstance(out, dict) else out

        w = torch.ones_like(mask)
        if self.canopy_weight > 0 and canopy is not None:
            w = w + self.canopy_weight * (canopy >= 0.5).float()
        if self.ucr_weight > 0 and under_canopy is not None:
            w = w + self.ucr_weight * (under_canopy >= 0.5).float()
        l_bce = F.binary_cross_entropy_with_logits(seg, mask, weight=w)
        l_dice = self.dice(seg, mask)
        l_cl = self.cldice(seg, mask)
        total = self.w_bce * l_bce + self.w_dice * l_dice + self.w_cldice * l_cl
        comp = {"bce": l_bce.item(), "dice": l_dice.item(), "cldice": l_cl.item()}

        if (self.orient_weight > 0 and isinstance(out, dict) and "orient" in out
                and orient_target is not None):
            l_or = self._orient_loss(out["orient"], orient_target, mask)
            total = total + self.orient_weight * l_or
            comp["orient"] = l_or.item()
        return total, comp
