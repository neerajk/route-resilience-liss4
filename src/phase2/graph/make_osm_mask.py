"""Dev helper — build an OSM road-mask GeoTIFF aligned to a reference raster.

Gives Phase 2 a runnable input WITHOUT the model (develop on OSM masks). Self-
contained (osmnx + rasterio) so phase2 stays decoupled from phase1. The real input
at integration is the model's pred_mask.tif (same format).

osmnx: Boeing (2017). rasterio.features.rasterize burns buffered road lines.
"""
from __future__ import annotations

from pathlib import Path


def build_osm_mask(ref_raster: str, aoi: str | None, out: str,
                   network_type: str = "drive", buffer_m: float = 4.0) -> str:
    """Fetch OSM roads for the AOI and rasterise onto the reference raster grid.

    ref_raster : any GeoTIFF on the target grid (e.g. a LISS-IV band) -> CRS/transform/shape.
    aoi        : road-fetch polygon shapefile (null => use the raster bounds).
    out        : output binary road-mask GeoTIFF path.
    """
    import numpy as np
    import osmnx as ox
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    with rasterio.open(ref_raster) as ref:
        crs, transform, W, H = ref.crs, ref.transform, ref.width, ref.height

    # AOI bbox in lon/lat for osmnx
    if aoi:
        import geopandas as gpd
        bbox = [float(x) for x in gpd.read_file(aoi).to_crs("EPSG:4326").total_bounds]
    else:
        w, s, e, n = array_bounds(H, W, transform)
        bbox = list(transform_bounds(crs, "EPSG:4326", w, s, e, n))

    print(f"[make_osm_mask] querying '{network_type}' roads for {[round(x,4) for x in bbox]} ...")
    g = ox.graph_from_polygon(box(*bbox), network_type=network_type)
    edges = ox.graph_to_gdfs(g, nodes=False).to_crs(crs)
    geoms = [geom.buffer(buffer_m) for geom in edges.geometry if geom is not None]
    print(f"[make_osm_mask] {len(geoms)} roads -> rasterising onto {W}x{H} @ {crs}")

    mask = (rasterize(((gm, 1) for gm in geoms), out_shape=(H, W), transform=transform,
                      fill=0, dtype="uint8") if geoms else np.zeros((H, W), "uint8"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                crs=crs, transform=transform, compress="deflate")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(mask, 1)
    print(f"[make_osm_mask] wrote {int(mask.sum())} road px -> {out}")
    return out
