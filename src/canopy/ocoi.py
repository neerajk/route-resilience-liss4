"""Overhead Canopy Occlusion Index (OCOI) — per OSM road segment.

DEFINITION (proposed metric)
----------------------------
For each OSM edge (segment) we sample points along it (Treepedia-style; sampling.py).
At each point we read the OVERHEAD canopy from a CHM raster (canopy HEIGHT, from
the CHMv2/DINOv3 pipeline) and, optionally, NDVI (vegetation GREENNESS). A point is
"occluded" when:

    occluded(point) = [ CHM(point) > chm_thresh ]  AND  [ ndvi is None OR NDVI(point) > ndvi_thresh ]

and the segment index is the fraction of its points that are occluded:

    OCOI(segment) = (# occluded points on segment) / (# points on segment)   ∈ [0, 1]

OCOI = 0 -> fully open road; OCOI = 1 -> road entirely under canopy.

WHY CHM **and** NDVI
--------------------
- CHM answers "is something TALL over the road?" (height).
- NDVI answers "is it VEGETATION, not a building/shadow?" (greenness;
  NDVI=(NIR-Red)/(NIR+Red), Rouse et al. 1974 — high for vegetation).
Combining them separates TREE canopy from BUILT occlusion. Because CHMv2 estimates
*tree* height specifically (Tolan et al., 2024), CHM alone is already a strong
signal; NDVI hardens it against tall non-vegetation.

WHY PER-SEGMENT
---------------
The segment (= a graph EDGE) is the unit BOTH papers share:
- Paper 1 stratifies Occlusion-Recall by OCOI (does recall fall as OCOI rises?).
- Paper 2 weights criticality by OCOI ("critical AND occluded" corridors = the
  worst single points of failure that are also hardest to map).

INPUT  : points_gdf (sampling.py), edges_gdf (sources/osm.py), CHM GeoTIFF,
         optional NDVI GeoTIFF.
OUTPUT : edges GeoDataFrame + columns: ocoi, mean_chm, [mean_ndvi], n_points.

References (see REFERENCES.md)
-----------------------------
- Rouse et al. (1974) NDVI. Tolan et al. (2024) canopy height from imagery.
- Li et al. (2015) Treepedia GVI (sampling lineage / GVI as a *validation* cross-check).
"""
from __future__ import annotations

from typing import Optional


def sample_raster_at_points(points_gdf, raster_path: str, window: int = 0):
    """Read raster band-1 values at point locations.

    window=0  -> exact pixel under each point (fast; fine at 5.8 m GSD).
    window>0  -> MEAN over a (2*window+1)^2 px box (captures overhanging canopy on
                 high-resolution CHM, where a tree edge can occlude the road while
                 the centerline pixel reads 0).
    Returns a float32 array aligned with points_gdf rows.
    """
    import numpy as np              # lazy
    import rasterio                 # lazy
    with rasterio.open(raster_path) as ds:
        pts = points_gdf.to_crs(ds.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        if window <= 0:
            return np.array([v[0] for v in ds.sample(coords)], dtype="float32")
        out = []
        for x, y in coords:
            r, c = ds.index(x, y)
            r0, r1 = max(0, r - window), min(ds.height, r + window + 1)
            c0, c1 = max(0, c - window), min(ds.width, c + window + 1)
            arr = ds.read(1, window=((r0, r1), (c0, c1)))
            out.append(float(np.nanmean(arr)) if arr.size else float("nan"))
        return np.array(out, dtype="float32")


def compute_ocoi(points_gdf, edges_gdf, chm_path: str,
                 ndvi_path: Optional[str] = None, chm_thresh: float = 3.0,
                 ndvi_thresh: float = 0.3, window: int = 1):
    """Compute per-segment OCOI. See module docstring for the exact definition."""
    import numpy as np              # lazy

    chm = sample_raster_at_points(points_gdf, chm_path, window=window)
    occ = chm > chm_thresh
    ndvi = None
    if ndvi_path:
        ndvi = sample_raster_at_points(points_gdf, ndvi_path, window=window)
        occ = occ & (ndvi > ndvi_thresh)

    pts = points_gdf.copy()
    pts["chm"] = chm
    pts["occluded"] = occ.astype("float32")
    if ndvi is not None:
        pts["ndvi"] = ndvi

    grouped = pts.groupby("edge_id")
    agg = grouped.agg(n_points=("occluded", "size"),
                      ocoi=("occluded", "mean"),
                      mean_chm=("chm", "mean"))
    if ndvi is not None:
        agg["mean_ndvi"] = grouped["ndvi"].mean()

    edges = edges_gdf.reset_index(drop=True).copy()
    # edge_id == positional index of edges after reset_index (matches sampling.py)
    edges = edges.merge(agg, left_index=True, right_index=True, how="left")
    edges["ocoi"] = edges["ocoi"].fillna(0.0)
    return edges
