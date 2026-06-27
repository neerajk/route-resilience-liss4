"""Phase I extras — augmentation for occlusion-robustness and scale-robustness.

Three components, composed in this order:
  1. OcclusionAugment  — CHM/canopy-DRIVEN synthetic occlusion. Where canopy is
     present we overwrite the road's spectral signature with a canopy-like one,
     synthetically HIDING roads. Training the model to still predict the
     (un-occluded) label under these conditions is the core occlusion-robustness
     mechanism (MASTER_PLAN §5.4). Pure NumPy — runs now.
  2. ScaleAugment      — random sensor-realistic GSD degradation via the MTF
     blur-downsample model (preprocess.degrade), for scale-robust features
     (cf. Scale-MAE, Reed et al., ICCV 2023). Pure NumPy — runs now.
  3. PhotometricGeometric — Albumentations flips/rotations/brightness, etc.
     OPTIONAL (lazy import). Albumentations: Buslaev et al. (2020), *Information*.
     Skipped with a warning if not installed.

A sample is a dict of NumPy arrays:
  image[C,H,W] (channel 0..2 = G,R,NIR; then ndvi, chm), mask[H,W], canopy[H,W].
Augments operate on NumPy BEFORE tensor conversion (see data/dataset.py).
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List

import numpy as np

from ..preprocess.degrade import degrade_then_restore


class OcclusionAugment:
    """Synthetically occlude road pixels that fall under canopy.

    p: probability of applying. extra_canopy_frac: optionally grow the canopy
    region to occlude MORE roads than the base tile (harder examples).
    """

    def __init__(self, p: float = 0.5, nir_boost: float = 0.5,
                 red_drop: float = 0.08, seed: int | None = None) -> None:
        self.p = p
        self.nir_boost = nir_boost
        self.red_drop = red_drop
        self.rng = np.random.default_rng(seed)

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.rng.random() > self.p:
            return s
        img = s["image"].copy()                        # avoid in-place mutation of cached arrays
        mask, canopy = s["mask"], s["canopy"]
        occ = (mask > 0.5) & (canopy > 0.5)            # road pixels under canopy
        if occ.any():
            # channels: 0=green,1=red,2=nir  -> push toward canopy signature
            img[2][occ] = np.clip(img[2][occ] + self.nir_boost, 0, 1)   # NIR up
            img[1][occ] = np.clip(img[1][occ] - self.red_drop, 0, 1)    # Red down
            # NOTE: label `mask` is UNCHANGED — the road is still there, just hidden.
        s["image"] = img
        return s


class ScaleAugment:
    """Randomly degrade input to a coarser GSD (then restore size).

    scale_range: multiplicative GSD factors to sample from, e.g. (1.0, 2.5)
    simulates seeing the same scene at up to 2.5x coarser resolution.
    Applied to the optical+ndvi channels (not the binary mask/canopy).
    """

    def __init__(self, p: float = 0.4, scale_range=(1.0, 2.5),
                 seed: int | None = None) -> None:
        self.p = p
        self.lo, self.hi = scale_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.rng.random() > self.p:
            return s
        factor = float(self.rng.uniform(self.lo, self.hi))
        if factor <= 1.0:
            return s
        img = s["image"]
        # degrade each channel-stack as [H,W,C] then restore
        chw = np.moveaxis(img, 0, -1)
        chw = degrade_then_restore(chw, factor, channel_axis=-1)
        s["image"] = np.moveaxis(chw, -1, 0).astype(np.float32)
        return s


class PhotometricGeometric:
    """Optional Albumentations pipeline (flips/rotate/brightness). Lazy + safe."""

    def __init__(self, p: float = 0.5) -> None:
        self._aug = None
        try:
            import albumentations as A            # lazy/optional
            self._aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ], additional_targets={"canopy": "mask"})
        except ImportError:
            warnings.warn("albumentations not installed; geometric/photometric "
                          "augmentation skipped. `pip install albumentations`.")

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._aug is None:
            return s
        img = np.moveaxis(s["image"], 0, -1)         # [H,W,C] for albumentations
        out = self._aug(image=img, mask=s["mask"], canopy=s["canopy"])
        s["image"] = np.moveaxis(out["image"], -1, 0).astype(np.float32)
        s["mask"] = out["mask"].astype(np.float32)
        s["canopy"] = out["canopy"].astype(np.float32)
        return s


class Compose:
    """Apply a list of augment callables in order."""

    def __init__(self, augs: List[Callable]) -> None:
        self.augs = augs

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for a in self.augs:
            s = a(s)
        return s


def build_augment(cfg: dict) -> Callable | None:
    """Construct the augmentation pipeline from cfg['augment'] (or None)."""
    a = cfg.get("augment", {})
    if not a.get("enabled", False):
        return None
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    augs: List[Callable] = []
    if a.get("occlusion", {}).get("enabled", True):
        augs.append(OcclusionAugment(p=a["occlusion"].get("p", 0.5), seed=seed))
    if a.get("scale", {}).get("enabled", True):
        augs.append(ScaleAugment(p=a["scale"].get("p", 0.4),
                                 scale_range=tuple(a["scale"].get("range", (1.0, 2.5))),
                                 seed=seed))
    if a.get("photometric_geometric", {}).get("enabled", True):
        augs.append(PhotometricGeometric(p=a["photometric_geometric"].get("p", 0.5)))
    return Compose(augs) if augs else None
