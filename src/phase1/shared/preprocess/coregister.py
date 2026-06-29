"""Co-registration / resampling onto a reference grid.

WHY (MASTER_PLAN §1.2, §5.1): the CHM (from ESRI ~0.6 m) and Sentinel-2 (10 m)
must be resampled onto the LISS-IV (5.8 m) pixel grid in a single project CRS
(EPSG:32643, UTM 43N for Bengaluru) BEFORE stacking. A misaligned CHM is a
useless occlusion prior — it would describe canopy that isn't over the road.

Choice of resampling:
  - continuous rasters (reflectance, NDVI, CHM): BILINEAR / CUBIC.
  - categorical/label rasters: NEAREST (never average class ids).

Built on rasterio.warp.reproject (GDAL). Lazy-imported.
"""
from __future__ import annotations

from typing import Tuple


def reproject_to_grid(src_array, src_crs, src_transform,
                      dst_crs: str, dst_transform, dst_shape: Tuple[int, int],
                      resampling: str = "bilinear"):
    """Reproject/resample `src_array` onto a target grid.

    Parameters
    ----------
    src_array : np.ndarray [H,W] or [bands,H,W]
    src_crs, src_transform : source georeferencing
    dst_crs : e.g. "EPSG:32643"
    dst_transform : affine.Affine of the reference (LISS-IV) grid
    dst_shape : (H, W) of the reference grid
    resampling : "bilinear"|"cubic"|"nearest" (use nearest for label rasters)
    """
    import numpy as np                                   # lazy
    from rasterio.warp import reproject, Resampling      # lazy
    rs = {"bilinear": Resampling.bilinear, "cubic": Resampling.cubic,
          "nearest": Resampling.nearest}[resampling]

    single = src_array.ndim == 2
    src = src_array[None] if single else src_array
    dst = np.zeros((src.shape[0], dst_shape[0], dst_shape[1]), dtype="float32")
    for b in range(src.shape[0]):
        reproject(source=src[b], destination=dst[b],
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs, resampling=rs)
    return dst[0] if single else dst
