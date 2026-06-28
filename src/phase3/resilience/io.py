"""Phase 3 I/O — load the Phase 2 graph; write criticality GeoJSON + CSV.

GraphML stores attributes as strings, so numeric fields are coerced back to float
on load. Node criticality is written as points (x,y from the graph) coloured by
betweenness — drop into QGIS as the criticality heatmap.
"""
from __future__ import annotations

from pathlib import Path


def load_graph(path: str):
    """Read graph.graphml -> NetworkX graph with numeric edge/node attrs."""
    import networkx as nx
    g = nx.read_graphml(path)
    for _u, _v, d in g.edges(data=True):
        for k in ("length_m", "travel_time_s", "speed_kph"):
            if k in d:
                d[k] = float(d[k])
    for _n, d in g.nodes(data=True):
        for k in ("x", "y"):
            if k in d:
                d[k] = float(d[k])
    return g


def write_node_criticality(graph, bc: dict, crs, path: str) -> None:
    """Nodes -> point GeoJSON with a `betweenness` attribute (criticality heatmap)."""
    import geopandas as gpd
    from shapely.geometry import Point
    geoms, recs = [], []
    for n, d in graph.nodes(data=True):
        if "x" in d and "y" in d:
            geoms.append(Point(float(d["x"]), float(d["y"])))
            recs.append({"node": str(n), "betweenness": float(bc.get(n, 0.0)),
                         "degree": int(graph.degree(n))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(recs, geometry=geoms, crs=crs).to_file(path, driver="GeoJSON")


def write_csv(rows, header, path: str) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        w.writerows(rows)
