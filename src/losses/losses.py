"""Loss functions for occlusion-robust road segmentation.

The combined objective (MASTER_PLAN §5.3) is:
    L = w_bce*BCE + w_dice*Dice + w_cldice*clDice
each term addressing a distinct failure mode:

  - BCE (binary cross-entropy): per-pixel calibration.
  - Dice: overlap under heavy class imbalance — roads are a tiny pixel fraction,
    so cross-entropy alone is dominated by background. (Milletari et al., 2016)
  - clDice (centerline Dice): a TOPOLOGY-preserving term. It measures overlap
    between the soft *skeletons* of prediction and ground truth, so it directly
    rewards CONNECTIVITY — exactly what downstream graph healing (Phase II) needs.
    This is the "connectivity loss" the problem statement asks for.
    (Shit et al., 2021, CVPR)

References
----------
- Milletari, F., Navab, N., Ahmadi, S. (2016). "V-Net: Fully Convolutional
  Neural Networks for Volumetric Medical Image Segmentation" (Dice loss). 3DV.
- Shit, S. et al. (2021). "clDice — a Novel Topology-Preserving Loss Function
  for Tubular Structure Segmentation." CVPR. https://arxiv.org/abs/2003.07311
  (soft-skeleton via iterated soft morphological erosion/dilation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Soft 2D erosion = min-filter, implemented as -maxpool(-x) (3x3)."""
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Soft 2D dilation = max-filter (3x3)."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable soft skeleton (Shit et al., 2021, Algorithm 1).

    Input x: probabilities in [0,1], shape [B,1,H,W]. Returns a soft skeleton in
    [0,1]. Iterates erosion/opening, accumulating the "skeleton residue".
    """
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class DiceLoss(nn.Module):
    """Soft Dice loss for a single foreground class (expects logits)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        prob = prob.flatten(1)
        target = target.flatten(1)
        inter = (prob * target).sum(1)
        denom = prob.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class SoftClDiceLoss(nn.Module):
    """centerline-Dice loss (Shit et al., 2021). Expects logits.

    clDice = harmonic-mean of:
      Tprec = |S_pred ∩ V_true| / |S_pred|   (precision of predicted skeleton)
      Tsens = |S_true ∩ V_pred| / |S_true|   (sensitivity of true skeleton)
    where S = soft skeleton, V = soft mask. Loss = 1 - clDice.
    """

    def __init__(self, iters: int = 10, smooth: float = 1.0) -> None:
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        skel_pred = soft_skeletonize(prob, self.iters)
        skel_true = soft_skeletonize(target, self.iters)
        tprec = (skel_pred * target).sum((1, 2, 3)) + self.smooth
        tprec = tprec / (skel_pred.sum((1, 2, 3)) + self.smooth)
        tsens = (skel_true * prob).sum((1, 2, 3)) + self.smooth
        tsens = tsens / (skel_true.sum((1, 2, 3)) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens)
        return 1.0 - cldice.mean()


class CombinedRoadLoss(nn.Module):
    """Weighted BCE + Dice + clDice. Weights come from config (loss.weights).

    canopy_weight (loss.canopy_weight): when > 0 and a canopy mask is supplied,
    the BCE term is reweighted per-pixel as ``1 + canopy_weight * canopy`` so that
    MISSING an occluded (under-canopy) road is penalised harder than missing an
    open one — operationalising "teach the model especially under trees" and
    directly targeting Occlusion-Recall. 0 => standard unweighted BCE.
    """

    def __init__(self, w_bce: float = 0.3, w_dice: float = 0.4, w_cldice: float = 0.3,
                 cldice_iters: int = 10, canopy_weight: float = 0.0) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_cldice = w_cldice
        self.canopy_weight = float(canopy_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.cldice = SoftClDiceLoss(iters=cldice_iters)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                canopy: torch.Tensor | None = None):
        if self.canopy_weight > 0.0 and canopy is not None:
            w = 1.0 + self.canopy_weight * (canopy >= 0.5).float()
            l_bce = F.binary_cross_entropy_with_logits(logits, target, weight=w)
        else:
            l_bce = self.bce(logits, target)
        l_dice = self.dice(logits, target)
        l_cl = self.cldice(logits, target)
        total = self.w_bce * l_bce + self.w_dice * l_dice + self.w_cldice * l_cl
        # return components too, for logging / publication loss curves
        return total, {"bce": l_bce.item(), "dice": l_dice.item(), "cldice": l_cl.item()}
