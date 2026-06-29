"""PyTorch Datasets for Phase I.

INPUT STACK (MASTER_PLAN §5.1): channels = [Green, Red, NIR, NDVI, CHM].
This 5-channel multimodal stack is the core design choice — RGB-only models are
"spectrally blind" under canopy, so we hand the network the CHM occlusion prior
and the NDVI canopy discriminator alongside the optical bands.

Two datasets:
  - SyntheticRoadDataset : on-the-fly fixtures (runs anywhere, no data needed).
  - TileFolderDataset    : reads pre-tiled real data from disk (LISS-IV path).

Each __getitem__ returns a dict of torch tensors:
  image  : float32 [C, H, W]   (C = len(channels), default 5)
  mask   : float32 [1, H, W]   road label in {0,1}
  canopy : float32 [1, H, W]   occluding-canopy mask (for Occlusion-Recall)

NORMALIZATION: synthetic tiles are already in [0,1] (norm=None => no-op). REAL
LISS-IV 10-bit DN MUST be standardised — pass per-channel (mean,std) via
cfg.data.norm so the network does not see raw 0-1023 DN. Indices (NDVI) should be
computed on reflectance, not DN (see data/indices.py).

References
----------
- Tolan et al. (2024), Remote Sensing of Environment (CHM from imagery).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import generate_tile

# Canonical channel order. Keep model `in_channels` in config in sync with this.
DEFAULT_CHANNELS: tuple = ("green", "red", "nir", "ndvi", "chm")

Norm = Optional[Tuple[np.ndarray, np.ndarray]]   # (mean[C], std[C]) or None


def _to_tensors(image: np.ndarray, mask: np.ndarray, canopy: np.ndarray,
                augment: Optional[Callable], norm: Norm = None) -> Dict[str, torch.Tensor]:
    """Apply augment (NumPy domain), optional per-channel normalize, then tensorise.

    Augments operate on {image[C,H,W], mask[H,W], canopy[H,W]} (see data/augment.py),
    so masks are 2-D here and promoted to [1,H,W] only at tensor conversion.
    """
    sample = {"image": image.astype(np.float32),
              "mask": mask.astype(np.float32),
              "canopy": canopy.astype(np.float32)}
    if augment is not None:
        sample = augment(sample)
    img = sample["image"]
    if norm is not None:                                  # standardise real DN
        mean, std = norm
        img = (img - mean.reshape(-1, 1, 1)) / (std.reshape(-1, 1, 1) + 1e-6)
        img = img.astype(np.float32)
    return {
        "image": torch.from_numpy(np.ascontiguousarray(img)),
        "mask": torch.from_numpy(np.ascontiguousarray(sample["mask"][None])),
        "canopy": torch.from_numpy(np.ascontiguousarray(sample["canopy"][None])),
    }


def _stack_channels(tile: Dict[str, np.ndarray], channels: Sequence[str]) -> np.ndarray:
    """Assemble the [C,H,W] image stack from a tile dict in channel order."""
    layers: List[np.ndarray] = []
    for ch in channels:
        if ch in ("green", "red", "nir"):
            idx = {"green": 0, "red": 1, "nir": 2}[ch]
            layers.append(tile["bands"][idx])
        elif ch == "ngrdi":
            # VISTA-v2 channel: (G-R)/(G+R), computed on-the-fly from bands so no
            # re-ingest is needed (NIR-free input). See data/indices.py:ngrdi.
            from .indices import ngrdi as _ngrdi
            layers.append(_ngrdi(tile["bands"][0], tile["bands"][1]))
        elif ch in tile:
            layers.append(tile[ch])
        else:
            raise KeyError(f"Channel '{ch}' not available in tile keys {list(tile)}")
    return np.stack(layers, axis=0).astype(np.float32)


class SyntheticRoadDataset(Dataset):
    """Procedurally generated tiles. Deterministic given a base seed.

    Parameters
    ----------
    length : int            number of tiles per epoch
    size : int              tile side length (pixels)
    channels : sequence     channel order (default 5-channel stack)
    canopy_fraction : float fraction of tile under occluding canopy
    n_roads, road_width : synthetic difficulty (config-driven)
    seed : int              base RNG seed (tile i uses seed+i)
    norm : (mean,std)|None  per-channel standardisation (usually None for synthetic)
    """

    def __init__(
        self,
        length: int = 64,
        size: int = 256,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        canopy_fraction: float = 0.35,
        n_roads: int = 9,
        road_width: int = 1,
        seed: int = 0,
        augment: Optional[Callable] = None,
        norm: Norm = None,
    ) -> None:
        self.length = int(length)
        self.size = int(size)
        self.channels = tuple(channels)
        self.canopy_fraction = float(canopy_fraction)
        self.n_roads = int(n_roads)
        self.road_width = int(road_width)
        self.seed = int(seed)
        self.augment = augment
        self.norm = norm

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile = generate_tile(
            size=self.size,
            n_roads=self.n_roads,
            road_width=self.road_width,
            canopy_fraction=self.canopy_fraction,
            seed=self.seed + idx,
        )
        image = _stack_channels(tile, self.channels)
        return _to_tensors(image, tile["mask"], tile["canopy"], self.augment, self.norm)


class TileFolderDataset(Dataset):
    """Reads pre-tiled real data saved as .npz with the synthetic schema.

    >>> # ====================  USER INPUT REQUIRED  ====================
    >>> # Point cfg.data.root at a folder of .npz tiles, each containing arrays:
    >>> #   bands [3,H,W] (G,R,NIR), ndvi [H,W], chm [H,W], mask [H,W], canopy [H,W]
    >>> # Produce these with the preprocessing step (src/preprocess/pipeline.py):
    >>> #   1) fetch LISS-IV via Bhoonidhi STAC, S2 via Planetary Computer,
    >>> #   2) compute NDVI, co-register + resample CHM to the 5.8 m grid,
    >>> #   3) rasterise OSM roads (osmnx, all_touched=True) into `mask`,
    >>> #   4) tile everything to `tile_size`.
    >>> # Set cfg.data.norm.{mean,std} (per-channel) for the real 10-bit DN.
    >>> # ===============================================================
    """

    def __init__(self, root: str, channels: Sequence[str] = DEFAULT_CHANNELS,
                 augment: Optional[Callable] = None, norm: Norm = None) -> None:
        self.root = Path(root)
        self.channels = tuple(channels)
        self.augment = augment
        self.norm = norm
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz tiles in {self.root}. See USER INPUT note in TileFolderDataset."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z = np.load(self.files[idx])
        tile = {k: z[k] for k in z.files}
        image = _stack_channels(tile, self.channels)
        # imagery-only tiles (no labels yet) -> zero mask/canopy so they still
        # load for EDA / inference; training needs real masks (write_mask=true).
        h, w = image.shape[1], image.shape[2]
        mask = tile["mask"] if "mask" in tile else np.zeros((h, w), "float32")
        canopy = tile["canopy"] if "canopy" in tile else np.zeros((h, w), "float32")
        return _to_tensors(image, mask, canopy, self.augment, self.norm)
