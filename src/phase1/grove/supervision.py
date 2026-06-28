"""GROVE Stage 1 — supervision targets: under-canopy road mask + orientation field.

Two per-tile targets the GROVE arm trains on (consumed from Stage 3/4 onward):

  under_canopy_road [H,W] {0,1}
      = OSM road `mask` AND `canopy` (NDVI>thresh). These are road pixels the
        OPTICAL image renders as vegetation — exactly the pixels GROVE must learn
        to predict despite seeing foliage. Used as a FOCAL supervision target.

  orient [2,H,W] float32  (sin2θ, cos2θ), zero off-road
      = per-road-pixel direction of travel. Roads are UNDIRECTED, so orientation
        is axial (modulo 180°): θ and θ+π are the same road. We therefore double
        the angle — encode (sin 2θ, cos 2θ) — so the 0°/180° wrap is removed and
        the target is a smooth unit vector (Batra et al., CVPR 2019). This field
        is the continuity carrier: it propagates road direction across canopy gaps.
        Decode with θ = ½·atan2(sin2θ, cos2θ).

The orientation field is rasterised from the OSM centerlines (same vectors that
produced `mask`), so it needs no new data — only the OSM edges + each tile's grid.

INPUT  : per tile — `mask`, `canopy` (from ingest_liss4); OSM edges (sources/osm).
OUTPUT : arrays added IN-PLACE to each tile .npz by build_supervision.py.

References (see REFERENCES.md)
-----------------------------
- Batra, Singh, Reddy, Khan, Chandra, Jawahar (2019). "Improved Road Connectivity
  by Joint Learning of Orientation and Segmentation." CVPR. — axial sin/cos
  orientation as an auxiliary task improves connectivity under occlusion.
- Rouse et al. (1974) NDVI — the canopy discriminator (see data/indices.py).
"""
from __future__ import annotations

from typing import Optional


# --------------------------------------------------------------------------- #
# under-canopy road target                                                    #
# --------------------------------------------------------------------------- #
def under_canopy_road(mask, canopy):
    """``mask AND canopy`` — road pixels that the optical image shows as canopy.

    Both inputs are [H,W] arrays in {0,1} (any numeric dtype; thresholded at 0.5).
    Returns float32 [H,W] in {0,1}.
    """
    import numpy as np                          # lazy
    m = (np.asarray(mask) > 0.5)
    c = (np.asarray(canopy) > 0.5)
    return (m & c).astype("float32")


# --------------------------------------------------------------------------- #
# orientation field (sin2θ / cos2θ) from OSM centerlines                       #
# --------------------------------------------------------------------------- #
def prepare_segments(edges_gdf, crs: str, buffer_m: float):
    """Explode road edges into straight 2-vertex pieces, each with its bearing.

    Returns a GeoDataFrame (metric ``crs``) with columns:
        geometry : the segment buffered by ``buffer_m`` (the road footprint to burn)
        sin2, cos2 : the axial-orientation encoding of the segment bearing θ
                     (sin 2θ, cos 2θ); constant along a straight piece.

    Exploding to 2-vertex pieces means a curved road gets a locally-correct
    bearing everywhere (not one bearing for the whole polyline).
    """
    import numpy as np                          # lazy
    import geopandas as gpd                      # lazy
    from shapely.geometry import LineString      # lazy

    g = edges_gdf.to_crs(crs)
    geoms, sins, coss = [], [], []
    for line in g.geometry:
        if line is None or line.is_empty:
            continue
        # MultiLineString -> iterate parts; LineString -> single part
        parts = getattr(line, "geoms", [line])
        for part in parts:
            xy = list(part.coords)
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                dx, dy = (x1 - x0), (y1 - y0)
                if dx == 0 and dy == 0:
                    continue
                theta = np.arctan2(dy, dx)            # bearing in [-π, π]
                seg = LineString([(x0, y0), (x1, y1)]).buffer(buffer_m)
                geoms.append(seg)
                sins.append(float(np.sin(2.0 * theta)))   # axial: double the angle
                coss.append(float(np.cos(2.0 * theta)))
    return gpd.GeoDataFrame({"geometry": geoms, "sin2": sins, "cos2": coss}, crs=crs)


def rasterize_orientation(seg_gdf, transform, out_shape):
    """Burn the (sin2θ, cos2θ) of buffered segments onto a tile grid.

    Returns float32 [2, H, W]: band 0 = sin2θ, band 1 = cos2θ, 0 off-road.
    Overlaps (junctions) are last-wins; downstream losses mask by the road mask,
    where orientation at multi-direction junctions is ambiguous anyway.
    """
    import numpy as np                          # lazy
    from rasterio.features import rasterize      # lazy

    H, W = out_shape
    if seg_gdf is None or len(seg_gdf) == 0:
        return np.zeros((2, H, W), dtype="float32")
    sin_band = rasterize(((geom, s) for geom, s in zip(seg_gdf.geometry, seg_gdf["sin2"])),
                         out_shape=out_shape, transform=transform, fill=0.0, dtype="float32")
    cos_band = rasterize(((geom, c) for geom, c in zip(seg_gdf.geometry, seg_gdf["cos2"])),
                         out_shape=out_shape, transform=transform, fill=0.0, dtype="float32")
    return np.stack([sin_band, cos_band]).astype("float32")


def orientation_to_angle_deg(sin2, cos2):
    """Decode (sin2θ, cos2θ) back to an axial angle in degrees [0, 180) — for viz."""
    import numpy as np                          # lazy
    theta = 0.5 * np.arctan2(sin2, cos2)         # [-π/2, π/2]
    deg = np.degrees(theta) % 180.0
    return deg


def orientation_for_tile(seg_gdf, bounds, shape, crs: str):
    """Convenience: subset segments to a tile bbox and rasterise its orientation.

    Parameters
    ----------
    seg_gdf : output of prepare_segments (buffered segments + sin2/cos2), metric crs.
    bounds  : (left, bottom, right, top) of the tile in the SAME metric ``crs``.
    shape   : (H, W) tile pixel shape.
    """
    from rasterio.transform import from_bounds   # lazy

    left, bottom, right, top = [float(v) for v in bounds]
    sub = seg_gdf.cx[left:right, bottom:top]      # bbox spatial filter (fast)
    transform = from_bounds(left, bottom, right, top, shape[1], shape[0])
    return rasterize_orientation(sub, transform, shape)
