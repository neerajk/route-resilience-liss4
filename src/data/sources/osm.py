"""OpenStreetMap adapter — road vectors -> raster labels AND a NetworkX graph.

DUAL ROLE (MASTER_PLAN §2):
  - Phase 1: rasterised OSM roads = automatic ground-truth masks (zero manual
    labelling). OSM-derived reference labels are the standard auto-annotation
    source for road extraction (e.g., used in SpaceNet/DeepGlobe lineage).
  - Phase 2: the SAME query yields a routable NetworkX graph (the baseline graph
    on which criticality/resilience runs WITHOUT any model — the paper firewall).

Built with OSMnx (Boeing, 2017, *Computers, Environment and Urban Systems*).
API note: OSMnx 2.x changed bbox argument order; we use `graph_from_polygon`
with a shapely box, which is stable across 1.x/2.x.

LABEL BUFFER: at 5.8 m GSD, roads are ~1 px wide. Buffer OSM centerlines by only
~3-6 m (≈1 px). Over-buffering at medium resolution destroys topology
(MASTER_PLAN §1.2).
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def fetch_osm_roads(bbox: List[float], network_type: str = "drive"):
    """Fetch the drivable road network within bbox=[minlon,minlat,maxlon,maxlat].

    Returns (graph, edges_gdf): a NetworkX MultiDiGraph and its edge GeoDataFrame
    (the line geometries used for rasterisation).
    """
    import osmnx as ox                          # lazy
    from shapely.geometry import box            # lazy
    minlon, minlat, maxlon, maxlat = bbox
    polygon = box(minlon, minlat, maxlon, maxlat)
    print(f"[osm] querying '{network_type}' network for bbox {[round(x, 4) for x in bbox]} "
          f"(this can take a while for a large AOI) ...")
    graph = ox.graph_from_polygon(polygon, network_type=network_type)
    edges = ox.graph_to_gdfs(graph, nodes=False)  # edge GeoDataFrame (EPSG:4326)
    print(f"[osm] fetched {graph.number_of_nodes()} nodes / {len(edges)} road edges")
    return graph, edges


def rasterize_roads(edges_gdf, transform, out_shape: Tuple[int, int],
                    crs: str = "EPSG:32643", buffer_m: float = 4.0):
    """Rasterise road centerlines to a binary mask on a target grid.

    Parameters
    ----------
    edges_gdf : GeoDataFrame of road LINESTRINGs (any CRS; reprojected to `crs`).
    transform : affine.Affine of the target raster (the LISS-IV tile grid).
    out_shape : (height, width) of the target raster.
    buffer_m  : half-width buffer in metres (~1 px at 5.8 m -> ~3-6 m).
    """
    from rasterio.features import rasterize     # lazy
    import numpy as np
    g = edges_gdf.to_crs(crs)
    geoms = [geom.buffer(buffer_m) for geom in g.geometry if geom is not None]
    print(f"[osm] rasterizing {len(geoms)} buffered roads (buffer={buffer_m} m) "
          f"onto {out_shape} grid @ {crs} ...")
    if not geoms:
        print("[osm] WARNING: no road geometries to rasterize -> empty mask.")
        return np.zeros(out_shape, dtype=np.uint8)
    mask = rasterize(((geom, 1) for geom in geoms), out_shape=out_shape,
                     transform=transform, fill=0, dtype="uint8")
    return mask
