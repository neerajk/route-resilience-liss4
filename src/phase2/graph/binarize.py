"""Step 2 — binarize + clean a road mask before skeletonization.

Skeletonization is noise-sensitive: specks become fake nodes and 1-px gaps cause
fragmentation. So we threshold, drop small blobs, and close tiny gaps. Pattern
follows the CRESI/SpaceNet refine step (threshold -> open/close -> skeletonize).
"""
from __future__ import annotations

import numpy as np


def clean_binary(mask: np.ndarray, threshold: float = 0.5,
                 min_object_size: int = 50, closing_radius: int = 2) -> np.ndarray:
    """Probability/binary mask -> clean boolean road mask.

    threshold       : prob -> binary cut (ignored if mask already 0/1).
    min_object_size : remove connected blobs smaller than this (px).
    closing_radius  : morphological closing to bridge 1-2 px gaps (0 disables).
    """
    from skimage.measure import label
    from skimage.morphology import closing, disk

    binary = mask > threshold
    if closing_radius and closing_radius > 0:
        binary = closing(binary, disk(closing_radius))
    if min_object_size and min_object_size > 0:
        # version-agnostic small-object removal (skimage renamed the param across
        # versions) — label components and drop those <= min_object_size pixels.
        lbl = label(binary)
        counts = np.bincount(lbl.ravel())
        keep = counts > int(min_object_size)
        keep[0] = False                       # background label
        binary = keep[lbl]
    return binary.astype(bool)
