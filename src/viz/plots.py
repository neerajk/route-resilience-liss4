"""Publication-grade plotting helpers.

Goal (MASTER_PLAN §9): every figure dumped here should be drop-in for a paper —
vector PDF + high-DPI PNG, readable fonts, colour-blind-safe palette, no chartjunk.
Matplotlib defaults are tuned once via set_pub_style().

Colour-blind-safe palette follows Wong, B. (2011). "Points of view: Color
blindness." Nature Methods 8, 441.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong (2011) colour-blind-safe palette
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def set_pub_style() -> None:
    """Apply a consistent, publication-ready matplotlib style."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=WONG),
        "legend.frameon": False,
        "image.cmap": "viridis",
    })


def save_fig(fig, out_dir, name: str) -> None:
    """Save a figure as BOTH .pdf (vector, for LaTeX) and .png (for slides)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def _stretch(a, p_lo: float = 2, p_hi: float = 98):
    import numpy as np
    a = a.astype("float32")
    lo, hi = np.percentile(a, [p_lo, p_hi])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def save_prediction_panel(image, mask, canopy, logits, out_dir, name: str = "prediction",
                          thr: float = 0.5) -> None:
    """Qualitative 4-panel figure for ONE sample (the paper's hero visual).

    [ LISS-IV FCC (NIR-R-G) | GT roads | prediction>thr | occlusion overlay ]
    The overlay colours under-canopy true road pixels: GREEN = recovered by the
    model, RED = missed — a direct visual read of Occlusion-Recall.

    Args take torch tensors for one example: image [C,H,W], mask [1,H,W] or [H,W],
    canopy [1,H,W] or [H,W], logits [1,H,W] or [1,1,H,W].
    """
    import numpy as np
    import torch

    img = image.detach().cpu().numpy()
    m = np.asarray(mask.detach().cpu().numpy()).squeeze()
    c = np.asarray(canopy.detach().cpu().numpy()).squeeze()
    prob = torch.sigmoid(logits.detach().float()).cpu().numpy().squeeze()
    pred = (prob >= thr).astype("float32")

    fcc = np.dstack([_stretch(img[2]), _stretch(img[1]), _stretch(img[0])])  # NIR,R,G
    occ_true = (m > 0.5) & (c > 0.5)
    overlay = fcc.copy() * 0.6
    overlay[occ_true & (pred > 0.5)] = [0.0, 1.0, 0.0]   # recovered under canopy
    overlay[occ_true & (pred <= 0.5)] = [1.0, 0.0, 0.0]  # missed under canopy

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(fcc); ax[0].set_title("LISS-IV FCC (NIR-R-G)")
    ax[1].imshow(m, cmap="gray"); ax[1].set_title("GT roads")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title(f"prediction > {thr}")
    ax[3].imshow(overlay); ax[3].set_title("occlusion overlay (G=recovered, R=missed)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    save_fig(fig, out_dir, name)
