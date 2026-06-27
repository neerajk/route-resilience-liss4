"""Phase 2 I/O — read a road-mask GeoTIFF; write graph (GraphML) + GeoJSON.

The mask is the Phase 1 -> Phase 2 contract: a georeferenced GeoTIFF (probability
or binary), single band [H,W], with CRS + affine transform. For dev, an OSM mask
(see make_osm_mask.py) has the same format. All heavy deps are lazy-imported so the
package imports without the geo stack present (graceful degradation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def read_mask(path: str) -> Tuple[np.ndarray, object, object]:
    """Read a road-mask GeoTIFF -> (array[H,W] float32, transform, crs).

    transform + crs carry the georeferencing that keeps the graph in real-world
    coordinates (Step 5). Reads band 1.
    """
    import rasterio
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        return arr, ds.transform, ds.crs


def write_graph(graph, path: str) -> None:
    """Write the NetworkX graph as GraphML (the Phase 3 contract).

    GraphML can't store arrays/geometries, so we drop the pixel 'pts'/'o' and
    keep scalar attributes (x, y, length_m, travel_time_s, healed)."""
    import networkx as nx
    g = graph.copy()
    for _, d in g.nodes(data=True):
        d.pop("o", None); d.pop("pts", None)
    for _, _, d in g.edges(data=True):
        d.pop("pts", None); d.pop("geometry", None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, path)


def write_geojson(graph, crs, path: str) -> None:
    """Write road edges as a LineString GeoJSON (QGIS / dashboard / interop)."""
    import geopandas as gpd
    geoms, recs = [], []
    for u, v, d in graph.edges(data=True):
        if d.get("geometry") is None:
            continue
        geoms.append(d["geometry"])
        recs.append({"u": int(u), "v": int(v),
                     "length_m": float(d.get("length_m", 0.0)),
                     "travel_time_s": float(d.get("travel_time_s", 0.0)),
                     "healed": bool(d.get("healed", False))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(recs, geometry=geoms, crs=crs).to_file(path, driver="GeoJSON")


def save_overlay(skeleton, graph_before, graph_after, out_dir, name="graph_overlay") -> None:
    """Quick before/after-healing visual: skeleton + healed bridges in red."""
    import matplotlib.pyplot as plt
    from ...common.viz import save_fig
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    for a, g, title in ((ax[0], graph_before, "extracted graph"),
                        (ax[1], graph_after, "after healing")):
        a.imshow(skeleton, cmap="gray_r")
        for u, v, d in g.edges(data=True):
            pts = d.get("pts")
            if pts is not None:
                a.plot(pts[:, 1], pts[:, 0], "-", lw=0.6, color="#0072B2")
            elif d.get("healed"):
                no, nv = g.nodes[u]["o"], g.nodes[v]["o"]
                a.plot([no[1], nv[1]], [no[0], nv[0]], "-", lw=1.2, color="#D55E00")
        a.set_title(title); a.axis("off")
    save_fig(fig, out_dir, name)
