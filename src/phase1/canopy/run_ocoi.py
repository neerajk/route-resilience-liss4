"""CLI: compute per-segment OCOI for an AOI and write GeoJSON/CSV/figures.

PIPELINE: OSM roads (osmnx) -> Treepedia-style points -> sample CHM(+NDVI) ->
per-segment OCOI -> save vector + table + a choropleth map + an OCOI histogram.

RUN (after env setup — see README §2):
    python -m src.phase1.canopy.run_ocoi --config config/phase1/config.yaml

NEEDS (USER INPUT in config.yaml):
  - preprocess.aoi_bbox      : [minlon, minlat, maxlon, maxlat]
  - preprocess.project_crs   : metric CRS, e.g. EPSG:32643
  - canopy.chm_path          : a CHM GeoTIFF (from your CHMv2/DINOv3 / openCHm run)
  - canopy.ndvi_path         : optional NDVI GeoTIFF (else CHM-only OCOI)

OUTPUT -> runs/canopy/<timestamp>/:
  ocoi_segments.geojson, ocoi_segments.csv, figures/ocoi_map.*, figures/ocoi_hist.*
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path


def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> Path:
    # lazy imports (full uv env)
    import matplotlib.pyplot as plt
    from ..data.sources import osm
    from ...common.viz import save_fig, set_pub_style
    from . import ocoi, sampling

    pp = cfg["preprocess"]
    c = cfg["canopy"]
    bbox = pp.get("aoi_bbox")
    crs = pp.get("project_crs", "EPSG:32643")
    if bbox is None:
        raise RuntimeError("Set preprocess.aoi_bbox in config.yaml (USER INPUT).")
    if not c.get("chm_path"):
        raise RuntimeError("Set canopy.chm_path to a CHM GeoTIFF (USER INPUT).")

    # 1. OSM roads -> 2. points -> 3. OCOI
    _, edges = osm.fetch_osm_roads(bbox, network_type=cfg["sources"]["osm"]["network_type"])
    points = sampling.sample_points_along_edges(edges, interval_m=c.get("interval_m", 20.0),
                                                metric_crs=crs)
    seg = ocoi.compute_ocoi(points, edges, chm_path=c["chm_path"],
                            ndvi_path=c.get("ndvi_path"),
                            chm_thresh=c.get("chm_thresh", 3.0),
                            ndvi_thresh=c.get("ndvi_thresh", 0.3),
                            window=c.get("window", 1))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg["paths"]["runs"]) / "canopy" / stamp
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # 4. save vector + table
    seg.to_crs(crs).to_file(out / "ocoi_segments.geojson", driver="GeoJSON")
    seg.drop(columns="geometry").to_csv(out / "ocoi_segments.csv", index=False)

    # 5. figures
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    seg.to_crs(crs).plot(column="ocoi", cmap="YlOrRd", linewidth=1.2, legend=True, ax=ax)
    ax.set_title("Per-segment Overhead Canopy Occlusion Index (OCOI)")
    ax.set_axis_off()
    save_fig(fig, out / "figures", "ocoi_map")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(seg["ocoi"].dropna(), bins=20)
    ax.set_xlabel("OCOI"); ax.set_ylabel("# segments")
    ax.set_title("Distribution of segment occlusion")
    save_fig(fig, out / "figures", "ocoi_hist")

    print(f"[OCOI] {len(seg)} segments | mean OCOI={seg['ocoi'].mean():.3f} -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-segment OCOI")
    ap.add_argument("--config", default="config/phase1/config.yaml")
    args = ap.parse_args()
    run(_load_config(args.config))


if __name__ == "__main__":
    main()
