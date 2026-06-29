"""Spectral indices derived from LISS-IV bands.

WHY THIS MODULE (MASTER_PLAN.md §1.2, §5.1): LISS-IV MX carries only
Green (B2), Red (B3), NIR (B4) — no Blue, no SWIR. NIR lets us compute NDVI
*natively* (no Sentinel-2 dependency), and NDVI is our occlusion discriminator:
canopy has high NDVI, impervious roads have low NDVI. So NDVI is appended to the
model input stack as a physically-motivated "where is canopy" prior.

INPUT  : numpy float arrays (reflectance), same HxW shape, for the needed bands.
OUTPUT : numpy float array, the index, same HxW shape.

References
----------
- NDVI: Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974).
  "Monitoring vegetation systems in the Great Plains with ERTS." NASA SP-351.
- SAVI (soil-adjusted, robust in sparse-veg/urban): Huete, A.R. (1988).
  "A soil-adjusted vegetation index (SAVI)." Remote Sensing of Environment 25(3).

NOTE ON RADIOMETRY: indices are only physically meaningful on reflectance
(ideally surface reflectance). On raw DN they are unreliable — see MASTER_PLAN
§1.1. Ensure Bhoonidhi products are at TOA/surface-reflectance level before use.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def ndvi(nir: np.ndarray, red: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red).

    Range roughly [-1, 1]; high => dense vegetation/canopy, low/negative =>
    impervious surfaces, bare soil, water. (Rouse et al., 1974)
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5, eps: float = _EPS) -> np.ndarray:
    """Soil-Adjusted Vegetation Index, less sensitive to soil background.

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L). L=0.5 is the canonical
    intermediate-cover value. Useful in mixed urban/sparse-vegetation scenes
    where NDVI saturates. (Huete, 1988)
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return ((nir - red) / (nir + red + L + eps)) * (1.0 + L)


def ngrdi(green: np.ndarray, red: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Normalized Green-Red Difference Index = (Green - Red) / (Green + Red).

    RGB-only greenness proxy (a.k.a. GRVI). Used by VISTA-v2 as the 3rd input
    channel so pretrain (DeepGlobe RGB) and fine-tune (LISS-IV G/R) share an
    IDENTICAL channel stack [G, R, NGRDI] with NO NIR — a domain-invariant input.
    Weaker canopy discriminator than NDVI (no NIR), and a deterministic function
    of G,R (adds no new information, only a ready-made prior + 3-channel shape).
    Range ~[-1,1]; higher over vegetation (G>R). (Tucker 1979; Motohka et al. 2010)
    """
    green = green.astype(np.float32)
    red = red.astype(np.float32)
    return (green - red) / (green + red + eps)
