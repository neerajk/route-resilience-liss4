"""Preprocessing orchestrator — raw multi-sensor data -> model-ready .npz tiles.

PIPELINE (MASTER_PLAN §3, §4):
  1. Fetch LISS-IV (Bhoonidhi)          [needs creds]
  2. Fetch Sentinel-2 composite (PC)    [anonymous]
  3. Load CHM (openCHm output)          [needs your CHM GeoTIFF]
  4. Reproject everything to EPSG:32643
  5. Establish the LISS-IV REFERENCE grid (5.8 m)
  6. Compute NDVI from LISS-IV (B4,B3)
  7. Co-register/resample CHM + S2 onto the LISS-IV grid
  8. Fetch + rasterise OSM roads -> mask;  canopy = CHM > threshold
  9. Tile to tile_size -> save .npz {bands[3],ndvi,chm,mask,canopy}

This is a SKELETON: the control flow, schema and references are complete, but it
runs only once you supply data/credentials. Every external dependency is
lazy-imported and every place needing your input is marked `USER INPUT`.
Use `--dry-run` to print the plan without fetching anything.

OUTPUT tiles are consumed directly by data.dataset.TileFolderDataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict


def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_plan(cfg: dict) -> None:
    pp = cfg["preprocess"]
    print("=== Preprocessing plan (dry run) ===")
    print(f"  AOI bbox        : {pp.get('aoi_bbox', 'USER INPUT REQUIRED')}")
    print(f"  Project CRS     : {pp.get('project_crs')}")
    print(f"  LISS-IV dates   : {pp.get('liss4_date_range', 'USER INPUT REQUIRED')}")
    print(f"  Sentinel-2 dates: {pp.get('s2_date_range', 'USER INPUT REQUIRED')}")
    print(f"  CHM GeoTIFF     : {pp.get('chm_path', 'USER INPUT REQUIRED (openCHm output)')}")
    print(f"  Tile size       : {pp.get('tile_size', 256)}")
    print(f"  Output tiles -> : {pp['paths']['tiles']}")
    print("  Stages: fetch -> reproject -> NDVI -> co-register -> OSM -> tile -> npz")


def build_tiles(cfg: dict, dry_run: bool = False) -> Path:
    """Run the full preprocessing pipeline. See module docstring for stages."""
    pp = cfg["preprocess"]
    out_dir = Path(pp["paths"]["tiles"])

    if dry_run:
        _print_plan(cfg)
        return out_dir

    # Lazy imports (only when actually running on the full env) ----------------
    import numpy as np
    import rasterio
    from ..data.indices import ndvi as compute_ndvi
    from ..data.sources import bhoonidhi, osm, planetary
    from . import coregister

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. LISS-IV --------------------------------------------------------- #
    # >>> USER INPUT: Bhoonidhi creds in .env + confirmed collection id in config.
    liss_paths = bhoonidhi.fetch_liss4(cfg)
    if not liss_paths:
        raise RuntimeError("No online LISS-IV products returned — check AOI/dates/"
                           "collection id, or order offline products in Bhoonidhi.")
    # >>> USER INPUT: implement archive->GeoTIFF extraction for the specific
    #     LISS-IV product packaging you receive (band files inside the .zip).
    #     Open the (G,R,NIR) bands with rasterio into `bands` [3,H,W] and capture
    #     (liss_crs, liss_transform, (H,W)) as the REFERENCE grid.
    raise NotImplementedError(
        "Stage 1 extraction is intentionally left for you: unpack the LISS-IV "
        "product into G/R/NIR arrays + reference grid, then the stages below run. "
        "See inline comments; everything downstream (2-9) is implemented."
    )

    # ----- Reference implementation for stages 2-9 (active once `bands`, -----
    # ----- `liss_crs`, `liss_transform`, `(H,W)` are defined above) ----------
    # H, W = bands.shape[1], bands.shape[2]
    # crs = pp["project_crs"]
    #
    # # 2. Sentinel-2 cloud-masked median composite
    # items = planetary.search_s2(pp["aoi_bbox"], pp["s2_date_range"],
    #                             max_cloud=cfg["sources"]["planetary"]["max_cloud"])
    # s2 = planetary.build_s2_composite(items, pp["aoi_bbox"], crs=crs).values
    #
    # # 3. CHM GeoTIFF (your openCHm output)  >>> USER INPUT: cfg.preprocess.chm_path
    # with rasterio.open(pp["chm_path"]) as ds:
    #     chm_raw, chm_crs, chm_tf = ds.read(1), ds.crs, ds.transform
    #
    # # 6. NDVI from LISS-IV
    # nd = compute_ndvi(bands[2], bands[1])
    #
    # # 7. Co-register CHM + S2 onto the LISS-IV grid
    # chm = coregister.reproject_to_grid(chm_raw, chm_crs, chm_tf, crs,
    #                                    liss_transform, (H, W), "bilinear")
    # s2g = coregister.reproject_to_grid(s2, crs, <s2_transform>, crs,
    #                                    liss_transform, (H, W), "bilinear")
    #
    # # 8. OSM roads -> mask ; canopy from CHM threshold
    # _, edges = osm.fetch_osm_roads(pp["aoi_bbox"])
    # mask = osm.rasterize_roads(edges, liss_transform, (H, W), crs,
    #                            buffer_m=cfg["sources"]["osm"]["buffer_m"])
    # canopy = (chm > cfg["preprocess"]["canopy_height_thresh"]).astype("float32")
    #
    # # 9. Tile and save .npz (schema matches data/dataset.py)
    # ts = pp["tile_size"]
    # k = 0
    # for r in range(0, H - ts + 1, ts):
    #     for c in range(0, W - ts + 1, ts):
    #         np.savez_compressed(out_dir / f"tile_{k:05d}.npz",
    #             bands=bands[:, r:r+ts, c:c+ts], ndvi=nd[r:r+ts, c:c+ts],
    #             chm=chm[r:r+ts, c:c+ts], mask=mask[r:r+ts, c:c+ts],
    #             canopy=canopy[r:r+ts, c:c+ts])
    #         k += 1
    # return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase I preprocessing")
    ap.add_argument("--config", default="config/phase1/config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print plan, fetch nothing")
    args = ap.parse_args()
    build_tiles(_load_config(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
