"""Step 5 — georeference the pixel graph into real-world coordinates.

Apply the raster's affine transform to every node/edge: world (x,y) =
T * (col + 0.5, row + 0.5). Each node gets x,y (CRS units, metres for UTM); each
edge gets a shapely LineString geometry + length_m. Assumes a PROJECTED CRS
(EPSG:32643 / UTM 43N here) so lengths are true metres.
"""
from __future__ import annotations

import numpy as np


def _pix_to_world(transform, rows, cols):
    """Vectorised pixel (row,col) -> world (x,y) at pixel centres."""
    from rasterio.transform import xy
    xs, ys = xy(transform, list(np.asarray(rows)), list(np.asarray(cols)), offset="center")
    return np.asarray(xs, "float64"), np.asarray(ys, "float64")


def georeference(graph, transform, crs):
    """Annotate `graph` in place with world coords + edge geometries/lengths.

    Nodes gain x,y. Edges gain geometry (LineString) and length_m. Returns graph."""
    from shapely.geometry import LineString

    # nodes
    for n, d in graph.nodes(data=True):
        r, c = float(d["o"][0]), float(d["o"][1])
        x, y = _pix_to_world(transform, [r], [c])
        d["x"], d["y"] = float(x[0]), float(y[0])

    # edges
    for u, v, d in graph.edges(data=True):
        pts = d.get("pts")
        if pts is not None and len(pts) >= 2:
            xs, ys = _pix_to_world(transform, pts[:, 0], pts[:, 1])
            coords = list(zip(xs.tolist(), ys.tolist()))
        else:  # degenerate edge: straight node-to-node line
            coords = [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                      (graph.nodes[v]["x"], graph.nodes[v]["y"])]
        line = LineString(coords)
        d["geometry"] = line
        d["length_m"] = float(line.length)   # planar length (metres in UTM)

    graph.graph["crs"] = str(crs)
    return graph
