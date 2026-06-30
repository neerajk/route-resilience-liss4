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


# DeepGlobe JPEGs decode as R,G,B in this fixed order.
_DG_SRC_IDX = {"red": 0, "green": 1, "blue": 2}

# Derived RGB indices a channel may request (computed from the [0,1] R,G,B bands).
# NGRDI = (G-R)/(G+R) is the NIR-free, domain-invariant analogue of NDVI used by the
# VISTA-v2 [green, red, ngrdi] stack (DeepGlobe has no NIR; NGRDI transfers to LISS-IV).
_DG_INDEX = {"ngrdi", "vari", "gli", "exg"}


def _rgb_index(name: str, r: np.ndarray, g: np.ndarray, b: np.ndarray,
               eps: float = 1e-6) -> np.ndarray:
    """Compute a derived RGB vegetation index from [0,1] R,G,B bands."""
    if name == "ngrdi":
        return (g - r) / (g + r + eps)                  # Normalized Green-Red Diff. Index
    if name == "vari":
        return (g - r) / (g + r - b + eps)              # Visible Atmospherically Resistant
    if name == "gli":
        return (2 * g - r - b) / (2 * g + r + b + eps)  # Green Leaf Index
    if name == "exg":
        return 2 * g - r - b                            # Excess Green
    raise KeyError(name)


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
    channels : output band ORDER, selected from DeepGlobe's R/G/B. Default
        ``("green", "red", "blue")`` — NOT plain RGB — so the two bands DeepGlobe
        shares with LISS-IV (Green, Red) land on the SAME first two positions as the
        LISS-IV stack ``[green, red, nir, ndvi]``. That way the warm-start's 3→4ch
        stem inflation copies Green→Green and Red→Red (instead of swapping them);
        DeepGlobe's Blue maps onto the LISS-IV NIR slot (the one unavoidable
        cross-sensor mismatch — DeepGlobe has no NIR), and NDVI is mean-initialised.
    augment, norm : same hooks as the other datasets (norm usually None: RGB∈[0,1]).
    """

    def __init__(self, root: str, tile_size: int = 256,
                 source_gsd_m: float = 0.5, target_gsd_m: float = 5.8,
                 sat_suffix: str = "_sat.jpg", mask_suffix: str = "_mask.png",
                 mtf_at_nyquist: float = 0.2, limit: int = 0,
                 channels: Sequence[str] = ("green", "red", "blue"),
                 augment: Optional[Callable] = None, norm: Norm = None) -> None:
        self.root = Path(root)
        self.ts = int(tile_size)
        self.scale = float(target_gsd_m) / float(source_gsd_m)
        self.mask_suffix = mask_suffix
        self.mtf = float(mtf_at_nyquist)
        self.channels = tuple(channels)
        unknown = [c for c in self.channels if c not in _DG_SRC_IDX and c not in _DG_INDEX]
        if unknown:
            raise ValueError(
                f"DeepGlobeDataset: unknown channel(s) {unknown}. "
                f"Raw bands: {list(_DG_SRC_IDX)}; derived indices: {sorted(_DG_INDEX)}."
            )
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
        sat = sat[..., :3] / 255.0                          # RGB in [0,1] (R,G,B order)

        m = _imread(mask_p).astype(np.float32)
        if m.ndim == 3:                                     # RGB mask -> luminance
            m = m[..., :3].mean(axis=2)
        road = (m > 127.5).astype(np.float32)               # white = road

        # degrade the full RGB to ~5.8 m, then build the requested channels (raw band
        # OR derived index) in order. Indices are computed AFTER the degrade so they
        # match the band resolution the model actually sees.
        rgb_ts = self._degrade_resize(sat)                  # [ts,ts,3] degraded R,G,B
        r, g, b = rgb_ts[..., 0], rgb_ts[..., 1], rgb_ts[..., 2]
        layers = [(r, g, b)[_DG_SRC_IDX[ch]] if ch in _DG_SRC_IDX else _rgb_index(ch, r, g, b)
                  for ch in self.channels]
        image = np.stack(layers, axis=0).astype(np.float32)  # [C,ts,ts]

        small_hw = blur_downsample(road[..., None], self.scale, self.mtf, channel_axis=-1)
        mask_ts = (resize(small_hw[..., 0], (self.ts, self.ts), order=0,
                          preserve_range=True, anti_aliasing=False) > 0.5).astype(np.float32)

        canopy = np.zeros((self.ts, self.ts), np.float32)   # no canopy concept here
        return _to_tensors(image, mask_ts, canopy, self.augment, self.norm)
