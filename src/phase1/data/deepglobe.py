"""DeepGlobe Road Extraction dataset → downsampled (0.5 m → 5.8 m) RGB tiles.

PURPOSE (Phase-I pretraining, MASTER_PLAN §1.3 / gameplan Stage A): DeepGlobe is
6,226 × 1024² aerial images at **0.5 m** RGB with road masks. LISS-IV is **5.8 m**
G/R/NIR. We close the *resolution* gap by degrading DeepGlobe to ~5.8 m with the
sensor-realistic blur-downsample model (``preprocess.degrade``) BEFORE pretraining,
so the encoder learns road shape at the target GSD. The *spectral* gap (RGB vs
G/R/NIR) is closed later by the warm-start stem-inflation in ``train.load_pretrained``
(pretrain in 3-ch RGB → inflate to [G,R,NIR,NDVI]).

LAYOUT (standard DeepGlobe; configurable suffixes):
    <root>/**/<id>_sat.jpg     RGB image (0.5 m)
    <root>/**/<id>_mask.png    road mask (white = road); images without a mask
                               (valid/test splits) are skipped.

Each __getitem__ returns the SAME dict schema as the other datasets via
``dataset._to_tensors``: image[3,ts,ts] (RGB in [0,1]), mask[1,ts,ts] in {0,1},
canopy[1,ts,ts] = zeros (DeepGlobe has no canopy; Occlusion-Recall is N/A here —
pretraining is monitored on relaxed-F1/IoU instead).
"""
from __future__ import annotations

import os

# Windows: torch (libiomp5md) and skimage/imageio image-IO (libomp) can load two
# OpenMP runtimes and abort with "OMP: Error #15". Allow the duplicate here (the
# standard, safe-in-practice workaround); overridable if the user set it already.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from .dataset import Norm, _to_tensors
from ..preprocess.degrade import blur_downsample


def _imread(path: Path) -> np.ndarray:
    """Read an image to a NumPy array (skimage backend; lazy import)."""
    from skimage.io import imread
    return np.asarray(imread(str(path)))


class DeepGlobeDataset(Dataset):
    """DeepGlobe tiles degraded to ``target_gsd_m`` then resized to ``tile_size``.

    Parameters
    ----------
    root : folder containing the DeepGlobe images (searched recursively).
    tile_size : output side length (match the LISS-IV training tile_size).
    source_gsd_m, target_gsd_m : degrade by ``target/source`` (0.5 → 5.8 ⇒ ×11.6).
    sat_suffix, mask_suffix : filename suffixes locating the pair.
    mtf_at_nyquist : sensor sharpness for the blur-downsample (lower = blurrier).
    limit : cap the number of pairs (0 = all; small values for quick tests).
    augment, norm : same hooks as the other datasets (norm usually None: RGB∈[0,1]).
    """

    def __init__(self, root: str, tile_size: int = 256,
                 source_gsd_m: float = 0.5, target_gsd_m: float = 5.8,
                 sat_suffix: str = "_sat.jpg", mask_suffix: str = "_mask.png",
                 mtf_at_nyquist: float = 0.2, limit: int = 0,
                 channels: Sequence[str] = ("red", "green", "blue"),
                 augment: Optional[Callable] = None, norm: Norm = None) -> None:
        self.root = Path(root)
        self.ts = int(tile_size)
        self.scale = float(target_gsd_m) / float(source_gsd_m)
        self.mask_suffix = mask_suffix
        self.mtf = float(mtf_at_nyquist)
        self.channels = tuple(channels)
        self.augment = augment
        self.norm = norm
        # pair each *_sat with its *_mask; skip images that have no mask (valid/test)
        sats = sorted(self.root.rglob(f"*{sat_suffix}"))
        self.pairs = []
        for s in sats:
            m = s.with_name(s.name[: -len(sat_suffix)] + mask_suffix)
            if m.exists():
                self.pairs.append((s, m))
        if limit and limit > 0:
            self.pairs = self.pairs[:limit]
        if not self.pairs:
            raise FileNotFoundError(
                f"No DeepGlobe *{sat_suffix}/*{mask_suffix} pairs under {self.root}. "
                f"Expected the standard layout <root>/**/<id>{sat_suffix} + <id>{mask_suffix}."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _degrade_resize(self, img_hwc: np.ndarray) -> np.ndarray:
        """Blur-downsample to ~target GSD, then bilinear-resize to tile_size."""
        from skimage.transform import resize
        small = blur_downsample(img_hwc, self.scale, self.mtf, channel_axis=-1)
        return resize(small, (self.ts, self.ts, img_hwc.shape[2]), order=1,
                      preserve_range=True, anti_aliasing=False).astype(np.float32)

    def __getitem__(self, idx: int):
        from skimage.transform import resize
        sat_p, mask_p = self.pairs[idx]
        sat = _imread(sat_p).astype(np.float32)
        if sat.ndim == 2:                                   # grayscale -> 3-ch
            sat = np.repeat(sat[..., None], 3, axis=2)
        sat = sat[..., :3] / 255.0                          # RGB in [0,1]

        m = _imread(mask_p).astype(np.float32)
        if m.ndim == 3:                                     # RGB mask -> luminance
            m = m[..., :3].mean(axis=2)
        road = (m > 127.5).astype(np.float32)               # white = road

        # degrade RGB to ~5.8 m; mask follows at the same small grid then to tile_size
        img_ts = self._degrade_resize(sat)                  # [ts,ts,3]
        small_hw = blur_downsample(road[..., None], self.scale, self.mtf, channel_axis=-1)
        mask_ts = (resize(small_hw[..., 0], (self.ts, self.ts), order=0,
                          preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        image = np.moveaxis(img_ts, -1, 0)                  # [3,ts,ts]
        canopy = np.zeros((self.ts, self.ts), np.float32)   # no canopy concept here
        return _to_tensors(image, mask_ts, canopy, self.augment, self.norm)
