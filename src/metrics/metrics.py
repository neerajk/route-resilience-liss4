"""Evaluation metrics for Phase I — including the two the problem statement and
MASTER_PLAN §9 call out specifically.

Standard
--------
- IoU (Jaccard) and Dice (F1): overlap of predicted vs true road pixels.

Problem-specific (the paper's headline metrics)
-----------------------------------------------
- Relaxed / buffered IoU & P/R/F1 (tolerance = 3-5 px): a predicted road pixel
  within `buffer` px of a true road counts as TP, and vice-versa. This avoids
  unfairly penalising sub-pixel alignment shifts that are unavoidable at 5.8 m
  GSD (MASTER_PLAN §1.2). Implemented via morphological dilation of the masks.
  This mirrors the "relaxation" used in road-network evaluation, e.g. the
  correctness/completeness buffer of Wiedemann et al. (1998) and the relaxed
  F1 used in DeepGlobe road extraction (Demir et al., 2018, CVPRW).
- Occlusion-Recall: recall computed ONLY over pixels flagged as canopy-occluded.
  This is the single most important number for the paper — it measures whether
  the model recovers roads *under* canopy, the core claim (MASTER_PLAN §5.5).

All functions take torch tensors [B,1,H,W]; `logits=True` applies sigmoid+thresh.

References
----------
- Wiedemann, C., Heipke, C., Mayer, H., Jamet, O. (1998). "Empirical evaluation
  of automatically extracted road axes." (buffer-based completeness/correctness)
- Demir, I. et al. (2018). "DeepGlobe 2018: A Challenge to Parse the Earth
  through Satellite Images." CVPRW (relaxed road F1).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from skimage.morphology import binary_dilation, disk


def _binarize(x: torch.Tensor, logits: bool, thr: float) -> torch.Tensor:
    if logits:
        x = torch.sigmoid(x)
    return (x >= thr).float()


def iou_score(pred: torch.Tensor, target: torch.Tensor, logits: bool = True,
              thr: float = 0.5, eps: float = 1e-6) -> float:
    p = _binarize(pred, logits, thr)
    t = (target >= 0.5).float()
    inter = (p * t).sum().item()
    union = p.sum().item() + t.sum().item() - inter
    return (inter + eps) / (union + eps)


def dice_score(pred: torch.Tensor, target: torch.Tensor, logits: bool = True,
               thr: float = 0.5, eps: float = 1e-6) -> float:
    p = _binarize(pred, logits, thr)
    t = (target >= 0.5).float()
    inter = (p * t).sum().item()
    return (2 * inter + eps) / (p.sum().item() + t.sum().item() + eps)


def relaxed_prf(pred: torch.Tensor, target: torch.Tensor, buffer: int = 3,
                logits: bool = True, thr: float = 0.5, eps: float = 1e-6) -> Dict[str, float]:
    """Buffered precision/recall/F1 with a `buffer`-px tolerance.

    precision = TP_pred / |pred|, where a predicted pixel is TP if within
                `buffer` px of any true road pixel (i.e. lies in dilated GT).
    recall    = TP_true / |true|, symmetric (true pixel near any prediction).
    """
    p = _binarize(pred, logits, thr).cpu().numpy().astype(bool)
    t = (target >= 0.5).cpu().numpy().astype(bool)
    se = disk(buffer)
    prec_list, rec_list = [], []
    for i in range(p.shape[0]):
        pi, ti = p[i, 0], t[i, 0]
        t_dil = binary_dilation(ti, se)
        p_dil = binary_dilation(pi, se)
        tp_p = np.logical_and(pi, t_dil).sum()
        tp_t = np.logical_and(ti, p_dil).sum()
        prec = (tp_p + eps) / (pi.sum() + eps)
        rec = (tp_t + eps) / (ti.sum() + eps)
        prec_list.append(prec)
        rec_list.append(rec)
    prec = float(np.mean(prec_list))
    rec = float(np.mean(rec_list))
    f1 = 2 * prec * rec / (prec + rec + eps)
    return {"relaxed_precision": prec, "relaxed_recall": rec, "relaxed_f1": f1}


def relaxed_iou(pred: torch.Tensor, target: torch.Tensor, buffer: int = 3,
                logits: bool = True, thr: float = 0.5, eps: float = 1e-6) -> float:
    """Buffered IoU: predicted pixels within `buffer` px of a true road count as
    intersection. Mean over the batch (Wiedemann 1998 / Demir 2018 lineage)."""
    p = _binarize(pred, logits, thr).cpu().numpy().astype(bool)
    t = (target >= 0.5).cpu().numpy().astype(bool)
    se = disk(buffer)
    vals = []
    for i in range(p.shape[0]):
        pi, ti = p[i, 0], t[i, 0]
        inter = np.logical_and(pi, binary_dilation(ti, se)).sum()
        union = pi.sum() + ti.sum() - np.logical_and(pi, ti).sum()
        vals.append((inter + eps) / (union + eps))
    return float(np.mean(vals)) if vals else 0.0


def pixel_counts(pred: torch.Tensor, target: torch.Tensor, canopy: torch.Tensor,
                 logits: bool = True, thr: float = 0.5) -> Dict[str, float]:
    """Raw TP/FP/FN (+ occluded TP/total) for GLOBAL accumulation across batches.

    Pooling these over the whole val set and computing IoU/Dice/Occlusion-Recall
    from the sums is unbiased; averaging per-batch ratios is not (it over-weights
    tiles with few road/occluded pixels). train._validate uses this."""
    p = _binarize(pred, logits, thr)
    t = (target >= 0.5).float()
    c = (canopy >= 0.5).float()
    occ_true = t * c
    return {
        "tp": (p * t).sum().item(),
        "fp": (p * (1 - t)).sum().item(),
        "fn": ((1 - p) * t).sum().item(),
        "occ_tp": (p * occ_true).sum().item(),
        "occ_total": occ_true.sum().item(),
    }


def occlusion_recall(pred: torch.Tensor, target: torch.Tensor, canopy: torch.Tensor,
                     logits: bool = True, thr: float = 0.5, eps: float = 1e-6) -> float:
    """Recall restricted to true road pixels that are under occluding canopy.

    The headline metric: of the roads HIDDEN by canopy, what fraction does the
    model still recover? (MASTER_PLAN §5.5)
    """
    p = _binarize(pred, logits, thr)
    t = (target >= 0.5).float()
    c = (canopy >= 0.5).float()
    occ_true = t * c                      # true road pixels under canopy
    recovered = p * occ_true              # of those, predicted positive
    return (recovered.sum().item() + eps) / (occ_true.sum().item() + eps)


def evaluate_batch(logits: torch.Tensor, target: torch.Tensor, canopy: torch.Tensor,
                   buffer: int = 3, thr: float = 0.5) -> Dict[str, float]:
    """Compute the full metric suite for one batch. Returns a flat dict.

    NOTE: iou/dice/occlusion_recall here are per-batch ratios (fine for a quick
    look). For unbiased HEADLINE numbers, accumulate `pixel_counts` globally
    across the val set — train._validate does this."""
    out = {
        "iou": iou_score(logits, target, thr=thr),
        "dice": dice_score(logits, target, thr=thr),
        "occlusion_recall": occlusion_recall(logits, target, canopy, thr=thr),
        "relaxed_iou": relaxed_iou(logits, target, buffer=buffer, thr=thr),
    }
    out.update(relaxed_prf(logits, target, buffer=buffer, thr=thr))
    return out
