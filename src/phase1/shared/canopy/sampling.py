"""Treepedia-style point sampling along the OSM street network.

WHAT IT IS
----------
Treepedia (MIT Senseable City Lab; Li et al., 2015) measures street greenery by
generating sample points at a FIXED INTERVAL along the OSM road network and taking
one observation per point (its `createPoints` step does exactly this spacing). We
reuse that SAMPLING DESIGN — systematic, equally-spaced points along each road
edge — but at each point we will read an OVERHEAD canopy raster (CHM/NDVI) instead
of a Street-View image (see ocoi.py).

WHY interval sampling along centerlines (vs averaging over rasterised road pixels)
---------------------------------------------------------------------------------
1. Density-controlled & road-width-independent: every segment is sampled at the
   same spatial rate, so the per-segment statistic is comparable everywhere.
2. Edge-native: each point carries its source edge id, so aggregation maps
   directly onto the routable graph's EDGES (the unit shared by both papers).
3. Protocol comparability: it follows the established Treepedia method, so the
   index is citable/reproducible rather than ad hoc.
(At 5.8 m GSD a road is ~1 px, so centerline points ≈ road footprint; for
high-resolution CHM use a small window in ocoi.py to catch overhanging canopy.)

INPUT  : edges_gdf (GeoDataFrame of road LINESTRINGs, any CRS) from sources/osm.py.
OUTPUT : GeoDataFrame of Points (metric CRS) with columns:
           geometry, edge_id (positional id of source edge), dist_m (along-edge).

References (see REFERENCES.md)
-----------------------------
- Li, X., Zhang, C., Li, W., Ricard, R., Meng, Q., Zhang, W. (2015). "Assessing
  street-level urban greenery using Google Street View and a modified green view
  index." Urban Forestry & Urban Greening 14(3):675-685.
- MIT Treepedia / Treepedia_Public (createPoints sampling).
"""
from __future__ import annotations


def sample_points_along_edges(edges_gdf, interval_m: float = 20.0,
                              metric_crs: str = "EPSG:32643"):
    """Generate points every `interval_m` metres along each road edge.

    Parameters
    ----------
    edges_gdf : GeoDataFrame of LINESTRING road geometries.
    interval_m : spacing between samples in metres (Treepedia default ~20 m; it
        is a free parameter — smaller = denser/finer OCOI, slower).
    metric_crs : a PROJECTED CRS in metres (EPSG:32643 = UTM 43N, Bengaluru) so
        `interval_m` and lengths are true metres, not degrees.

    Returns
    -------
    GeoDataFrame (Point, metric_crs) with columns: edge_id, dist_m.
    """
    import geopandas as gpd          # lazy
    import numpy as np               # lazy
    from shapely.geometry import Point

    g = edges_gdf.to_crs(metric_crs).reset_index(drop=True)
    rows = []
    for eid, geom in enumerate(g.geometry):
        if geom is None or geom.length == 0:
            continue
        # include both endpoints; n+1 evenly spaced points
        n = max(1, int(np.floor(geom.length / interval_m)))
        for d in np.linspace(0.0, geom.length, n + 1):
            p = geom.interpolate(float(d))
            rows.append({"geometry": Point(p.x, p.y), "edge_id": int(eid),
                         "dist_m": float(d)})
    return gpd.GeoDataFrame(rows, crs=metric_crs)
