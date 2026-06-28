"""Step 3 — skeletonize a clean binary road mask to 1-px centerlines.

Morphological thinning peels boundary pixels while preserving connectivity until
each road is 1-pixel wide (its medial line). scikit-image `skeletonize`.
"""
from __future__ import annotations

import numpy as np


def skeletonize_mask(binary: np.ndarray) -> np.ndarray:
    """Boolean road mask -> boolean 1-px skeleton."""
    from skimage.morphology import skeletonize
    return skeletonize(binary.astype(bool)).astype(bool)
