"""Synthetic LISS-IV-like tiles with controllable occlusion.

WHY THIS EXISTS: it lets the *entire* Phase I pipeline (EDA, training, metrics,
visualisation) run on your M1 today with zero external data or API keys, and it
gives a controlled testbed where we KNOW the ground truth and the occluded
fraction — useful for unit-testing Occlusion-Recall before real LISS-IV arrives.
This is a DEVELOPMENT fixture, NOT training data for the paper.

The generator mimics the relevant physics qualitatively (not radiometrically):
  - Roads are thin, low-NIR, low-NDVI linear features (impervious).
  - Vegetation/canopy is high-NIR, high-NDVI.
  - A canopy layer (the CHM proxy) partially OCCLUDES roads by overwriting the
    road's spectral signature where canopy is present — exactly the failure mode
    Phase I must overcome (MASTER_PLAN §0, §5).

OUTPUT (per tile): dict with
  bands  : float32 [3, H, W]  -> (Green, Red, NIR) in [0,1]
  ndvi   : float32 [H, W]
  chm    : float32 [H, W]     -> canopy height proxy in [0,1]
  mask   : float32 [H, W]     -> road label {0,1} (the TRUE, un-occluded roads)
  canopy : float32 [H, W]     -> {0,1} where canopy occludes (for Occlusion-Recall)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.draw import line as sk_line
from skimage.morphology import dilation, disk

from .indices import ndvi as compute_ndvi


def generate_tile(
    size: int = 256,
    n_roads: int = 9,
    road_width: int = 1,
    canopy_fraction: float = 0.35,
    occlude: bool = True,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Generate one synthetic tile. See module docstring for output schema."""
    rng = np.random.default_rng(seed)

    # --- road skeleton: random straight segments between border points -------
    road = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_roads):
        r0, c0 = rng.integers(0, size), rng.integers(0, size)
        r1, c1 = rng.integers(0, size), rng.integers(0, size)
        rr, cc = sk_line(int(r0), int(c0), int(r1), int(c1))
        road[rr, cc] = 1
    if road_width > 1:
        road = dilation(road, disk(road_width // 2))
    mask = road.astype(np.float32)

    # --- canopy field: smooth random blobs (CHM proxy in [0,1]) --------------
    noise = rng.normal(size=(size, size)).astype(np.float32)
    # cheap smoothing via repeated box blur (keeps deps minimal)
    for _ in range(6):
        noise = (
            noise
            + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)
        ) / 5.0
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)  # np.ptp: numpy-2 safe
    thr = np.quantile(noise, 1.0 - canopy_fraction)
    canopy = (noise >= thr).astype(np.float32)
    chm = (noise * canopy).astype(np.float32)  # height proxy, 0 outside canopy

    # --- spectral bands (Green, Red, NIR) in [0,1] ---------------------------
    # Base: vegetation-ish background = high NIR; soil/built = moderate.
    green = 0.25 + 0.10 * rng.random((size, size)).astype(np.float32)
    red = 0.22 + 0.10 * rng.random((size, size)).astype(np.float32)
    nir = 0.55 + 0.20 * rng.random((size, size)).astype(np.float32)

    # Canopy: push NIR up, Red down (classic vegetation signature).
    nir = np.where(canopy > 0, nir + 0.20, nir)
    red = np.where(canopy > 0, red - 0.08, red)

    # Roads (impervious): low NIR, higher red, flat green.
    road_bool = mask > 0
    nir[road_bool] = 0.18 + 0.05 * rng.random(road_bool.sum()).astype(np.float32)
    red[road_bool] = 0.30 + 0.05 * rng.random(road_bool.sum()).astype(np.float32)
    green[road_bool] = 0.28 + 0.05 * rng.random(road_bool.sum()).astype(np.float32)

    # --- OCCLUSION: where canopy overlaps a road, canopy wins spectrally -----
    occluded = ((mask > 0) & (canopy > 0)).astype(np.float32)
    if occlude:
        ov = occluded > 0
        nir[ov] = 0.70 + 0.10 * rng.random(int(ov.sum())).astype(np.float32)
        red[ov] = 0.16 + 0.05 * rng.random(int(ov.sum())).astype(np.float32)
        green[ov] = 0.30 + 0.05 * rng.random(int(ov.sum())).astype(np.float32)

    bands = np.clip(np.stack([green, red, nir], axis=0), 0.0, 1.0).astype(np.float32)
    nd = compute_ndvi(bands[2], bands[1]).astype(np.float32)

    return {
        "bands": bands,          # [3,H,W] G,R,NIR
        "ndvi": nd,              # [H,W]
        "chm": chm,              # [H,W]
        "mask": mask,            # [H,W] true roads
        "canopy": canopy,        # [H,W] occluding canopy
    }
