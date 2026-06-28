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


class RadiometricJitter:
    """Independent per-band radiometric variation on the optical/NDVI channels.

    Samples a per-channel gain ~U(1-gain, 1+gain) and bias ~U(-bias, bias) (in the
    input's own units, applied BEFORE normalisation), plus optional Gaussian noise
    at ``noise``×(per-band std). Mask/canopy are untouched. Gain+bias is range-safe
    for both 10-bit DN and reflectance, so no [0,1] assumption is made.
    """

    def __init__(self, p: float = 0.5, gain: float = 0.1, bias: float = 0.05,
                 noise: float = 0.0, seed: int | None = None) -> None:
        self.p, self.gain, self.bias, self.noise = p, gain, bias, noise
        self.rng = np.random.default_rng(seed)

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.rng.random() > self.p:
            return s
        img = s["image"].copy()
        for ch in range(img.shape[0]):
            band = img[ch]
            g = 1.0 + self.rng.uniform(-self.gain, self.gain)
            b = self.rng.uniform(-self.bias, self.bias) * (float(np.std(band)) + 1e-6)
            band = band * g + b
            if self.noise > 0:
                band = band + self.rng.normal(0, self.noise * (float(np.std(band)) + 1e-6), band.shape)
            img[ch] = band
        s["image"] = img.astype(np.float32)
        return s


class RoadCoarseDropout:
    """Hide rectangular patches of the INPUT (label kept) to force gap-bridging.

    Unlike vanilla CoarseDropout, patch centres are biased onto road pixels
    (``road_bias``) so the holes preferentially SEVER roads — directly training
    occlusion recovery. Holes are filled with the per-channel tile mean (a neutral,
    in-distribution value). The road label under the hole is intentionally kept.
    """

    def __init__(self, p: float = 0.5, max_holes: int = 6, min_frac: float = 0.03,
                 max_frac: float = 0.12, road_bias: float = 0.8, seed: int | None = None) -> None:
        self.p, self.max_holes = p, max_holes
        self.min_frac, self.max_frac, self.road_bias = min_frac, max_frac, road_bias
        self.rng = np.random.default_rng(seed)

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.rng.random() > self.p:
            return s
        img = s["image"].copy()
        mask = s["mask"]
        C, H, W = img.shape
        fill = img.reshape(C, -1).mean(axis=1)
        road_px = np.argwhere(mask > 0.5)
        lo_h, hi_h = max(2, int(self.min_frac * H)), max(3, int(self.max_frac * H))
        lo_w, hi_w = max(2, int(self.min_frac * W)), max(3, int(self.max_frac * W))
        for _ in range(int(self.rng.integers(1, self.max_holes + 1))):
            hh = int(self.rng.integers(lo_h, hi_h + 1))
            ww = int(self.rng.integers(lo_w, hi_w + 1))
            if len(road_px) and self.rng.random() < self.road_bias:
                cy, cx = road_px[int(self.rng.integers(len(road_px)))]
            else:
                cy, cx = int(self.rng.integers(0, H)), int(self.rng.integers(0, W))
            y0, x0 = max(0, cy - hh // 2), max(0, cx - ww // 2)
            y1, x1 = min(H, y0 + hh), min(W, x0 + ww)
            img[:, y0:y1, x0:x1] = fill[:, None, None]
        s["image"] = img.astype(np.float32)          # mask unchanged: road still labelled
        return s


class CopyPasteRoads:
    """Intra-image road copy-paste: clone a road-containing window to a new offset.

    Picks a window around a random road pixel, then pastes its road pixels (image +
    canopy + label) at a random destination — OR-ing the mask. Increases road
    density and junction variety without needing a second sample (cf. Ghiasi et al.,
    2021, "Simple Copy-Paste"). Only road-labelled source pixels are written, so the
    background is preserved.
    """

    def __init__(self, p: float = 0.3, max_patches: int = 2, min_frac: float = 0.12,
                 max_frac: float = 0.33, seed: int | None = None) -> None:
        self.p, self.max_patches = p, max_patches
        self.min_frac, self.max_frac = min_frac, max_frac
        self.rng = np.random.default_rng(seed)

    def __call__(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.rng.random() > self.p:
            return s
        mask = s["mask"]
        road_px = np.argwhere(mask > 0.5)
        if len(road_px) == 0:
            return s
        img, m, can = s["image"].copy(), mask.copy(), s["canopy"].copy()
        C, H, W = img.shape
        for _ in range(int(self.rng.integers(1, self.max_patches + 1))):
            ph = int(self.rng.integers(max(4, int(self.min_frac * H)), max(5, int(self.max_frac * H)) + 1))
            pw = int(self.rng.integers(max(4, int(self.min_frac * W)), max(5, int(self.max_frac * W)) + 1))
            ph, pw = min(ph, H), min(pw, W)
            cy, cx = road_px[int(self.rng.integers(len(road_px)))]
            sy0 = int(np.clip(cy - ph // 2, 0, H - ph)); sx0 = int(np.clip(cx - pw // 2, 0, W - pw))
            dy0 = int(self.rng.integers(0, H - ph + 1)); dx0 = int(self.rng.integers(0, W - pw + 1))
            src_road = m[sy0:sy0 + ph, sx0:sx0 + pw] > 0.5      # paste only road pixels
            if not src_road.any():
                continue
            dst_img = img[:, dy0:dy0 + ph, dx0:dx0 + pw]
            dst_img[:, src_road] = img[:, sy0:sy0 + ph, sx0:sx0 + pw][:, src_road]
            m[dy0:dy0 + ph, dx0:dx0 + pw][src_road] = 1.0
            can[dy0:dy0 + ph, dx0:dx0 + pw][src_road] = \
                can[sy0:sy0 + ph, sx0:sx0 + pw][src_road]
        s["image"], s["mask"], s["canopy"] = img.astype(np.float32), m.astype(np.float32), can.astype(np.float32)
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
    # Order: structural (copy-paste) -> occlusion/dropout (hide input, keep label)
    # -> scale -> radiometric -> geometric/photometric. Each stage reseeds off the
    # base seed so stages don't share an RNG stream.
    cp = a.get("copy_paste", {})
    if cp.get("enabled", False):
        augs.append(CopyPasteRoads(p=cp.get("p", 0.3), max_patches=int(cp.get("max_patches", 2)),
                                   seed=seed + 1))
    if a.get("occlusion", {}).get("enabled", True):
        augs.append(OcclusionAugment(p=a["occlusion"].get("p", 0.5), seed=seed + 2))
    cd = a.get("coarse_dropout", {})
    if cd.get("enabled", True):
        augs.append(RoadCoarseDropout(p=cd.get("p", 0.5), max_holes=int(cd.get("max_holes", 6)),
                                      max_frac=float(cd.get("max_frac", 0.12)),
                                      road_bias=float(cd.get("road_bias", 0.8)), seed=seed + 3))
    if a.get("scale", {}).get("enabled", True):
        augs.append(ScaleAugment(p=a["scale"].get("p", 0.4),
                                 scale_range=tuple(a["scale"].get("range", (1.0, 2.5))),
                                 seed=seed + 4))
    rj = a.get("radiometric", {})
    if rj.get("enabled", True):
        augs.append(RadiometricJitter(p=rj.get("p", 0.5), gain=float(rj.get("gain", 0.1)),
                                      bias=float(rj.get("bias", 0.05)),
                                      noise=float(rj.get("noise", 0.0)), seed=seed + 5))
    if a.get("photometric_geometric", {}).get("enabled", True):
        augs.append(PhotometricGeometric(p=a["photometric_geometric"].get("p", 0.5)))
    return Compose(augs) if augs else None
