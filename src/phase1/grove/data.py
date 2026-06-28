"""GROVE dataset — VISTA tiles + the Stage-1 orientation target.

Wraps the shared tile loader and additionally returns the per-tile `orient`
[2,H,W] (sin2θ,cos2θ) array written by build_supervision. The `under_canopy_road`
target is NOT returned here — the trainer recomputes it from (mask ∧ canopy) AFTER
augmentation so it always stays in sync with the augmented mask.

CAVEAT (geometric augmentation): flips/rotations change road direction, so the
sin2θ/cos2θ field would need the same transform applied. That is not yet wired —
run GROVE multi-task training with geometric augmentation OFF (photometric/occlusion
only), or orientation supervision will be mislabelled. Seg-only runs are unaffected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from .supervision import under_canopy_road  # noqa: F401  (kept for parity/import use)
from ..shared.data.dataset import (DEFAULT_CHANNELS, Norm, TileFolderDataset,
                                   _stack_channels, _to_tensors)


class GroveTileDataset(TileFolderDataset):
    """TileFolderDataset + `orient` [2,H,W] target (zeros if a tile lacks it)."""

    def __init__(self, root: str, channels: Sequence[str] = DEFAULT_CHANNELS,
                 augment: Optional[Callable] = None, norm: Norm = None) -> None:
        super().__init__(root=root, channels=channels, augment=augment, norm=norm)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z = np.load(self.files[idx])
        tile = {k: z[k] for k in z.files}
        image = _stack_channels(tile, self.channels)
        h, w = image.shape[1], image.shape[2]
        mask = tile["mask"] if "mask" in tile else np.zeros((h, w), "float32")
        canopy = tile["canopy"] if "canopy" in tile else np.zeros((h, w), "float32")
        out = _to_tensors(image, mask, canopy, self.augment, self.norm)
        orient = tile["orient"] if "orient" in tile else np.zeros((2, h, w), "float32")
        out["orient"] = torch.from_numpy(np.ascontiguousarray(orient.astype("float32")))
        return out
