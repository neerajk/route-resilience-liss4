"""GROVE Stage 1 CLI — add under-canopy-road + orientation targets to existing tiles.

Reads the .npz tiles you ALREADY ingested (no re-ingest needed), fetches the OSM
road vectors once, and writes two new arrays INTO each tile (idempotent — re-running
overwrites just these keys, never touches bands/mask/canopy):

  under_canopy_road [H,W]  = mask AND canopy        (focal supervision target)
  orient            [2,H,W]= (sin2θ, cos2θ) per road pixel  (continuity carrier)

It is non-destructive and safe to re-run. It does NOT train anything (you run the
training yourself later, once the GROVE backbone lands in Stage 2).

RUN (after activating the env):
    python -m src.phase1.grove.build_supervision --config config/phase1/grove.yaml

Reads (via cfg, resolved through `extends:`):
    data.liss4.{green|stack, aoi, network_type}   -> reference grid CRS + OSM fetch
    data.root  (or grove.supervision.out_tiles)   -> folder of .npz tiles to augment
    grove.supervision.orientation.buffer_m        -> road half-width for orientation
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from ...common.config import load_config
from ..shared.preprocess.ingest_liss4 import _aoi_bbox, _pbar, _resolve_band_paths
from . import supervision as sv


def _ref_crs(lc: dict):
    """Reference-grid CRS from the LISS-IV band/stack GeoTIFF (tiles' `bounds` CRS)."""
    import rasterio                              # lazy
    stack, gp, _, _ = _resolve_band_paths(lc)
    ref_path = stack if stack else gp
    with rasterio.open(ref_path) as ref:
        return ref.crs, ref.transform, ref.width, ref.height


def build(cfg: dict) -> Path:
    lc = cfg.get("data", {}).get("liss4", {})
    if not lc:
        raise RuntimeError("cfg.data.liss4 block missing — see config/phase1/config.yaml.")
    gcfg = (cfg.get("grove", {}) or {}).get("supervision", {}) or {}
    ocfg = gcfg.get("orientation", {}) or {}
    do_orient = bool(ocfg.get("enabled", True))
    buffer_m = float(ocfg.get("buffer_m", lc.get("buffer_m", 4.0)))

    root = Path(gcfg.get("out_tiles") or cfg.get("data", {}).get("root", "data/tiles"))
    tiles = sorted(root.glob("*.npz"))
    if not tiles:
        raise RuntimeError(f"No .npz tiles in {root}. Run ingest_liss4 first.")

    # OSM vectors + reference CRS (only needed for the orientation field)
    seg_gdf = None
    crs = None
    if do_orient:
        from ..shared.data.sources import osm
        ref_crs, ref_tf, W, H = _ref_crs(lc)
        crs = str(ref_crs)
        bbox = _aoi_bbox(lc.get("aoi"), ref_crs, ref_tf, H, W)
        print(f"[grove] fetching OSM '{lc.get('network_type', 'drive')}' roads for orientation ...")
        _, edges = osm.fetch_osm_roads(bbox, network_type=str(lc.get("network_type", "drive")))
        seg_gdf = sv.prepare_segments(edges, crs=crs, buffer_m=buffer_m)
        print(f"[grove] prepared {len(seg_gdf)} oriented road segments (buffer={buffer_m} m) @ {crs}")

    n_done = n_skip = 0
    ucr_px = road_px = orient_px = 0
    bar = _pbar(tiles, desc="supervision", unit="tile")
    for f in bar:
        z = np.load(f)
        tile = {k: z[k] for k in z.files}
        if "mask" not in tile or "canopy" not in tile:
            n_skip += 1                               # imagery-only tile, no labels yet
            continue

        ucr = sv.under_canopy_road(tile["mask"], tile["canopy"])
        tile["under_canopy_road"] = ucr
        ucr_px += int(ucr.sum()); road_px += int((tile["mask"] > 0.5).sum())

        if do_orient:
            h, w = tile["mask"].shape[-2:]
            bounds = tile["bounds"] if "bounds" in tile else None
            if bounds is None:
                # no georef -> can't place the OSM orientation; write zeros (graceful)
                tile["orient"] = np.zeros((2, h, w), "float32")
            else:
                orient = sv.orientation_for_tile(seg_gdf, bounds, (h, w), crs)
                # confine orientation to actual road pixels (drop buffer bleed off-road)
                roadm = (tile["mask"] > 0.5).astype("float32")
                tile["orient"] = (orient * roadm[None]).astype("float32")
                orient_px += int((np.abs(tile["orient"]).sum(axis=0) > 0).sum())

        # atomic in-place rewrite (temp -> replace) so a crash can't corrupt a tile.
        # tmp ends in .npz, so np.savez_compressed writes to exactly that path.
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(root))
        os.close(fd)
        np.savez_compressed(tmp, **tile)
        os.replace(tmp, f)
        n_done += 1
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(done=n_done, skipped=n_skip)

    pct = (100.0 * ucr_px / road_px) if road_px else 0.0
    print(f"[grove] DONE: augmented {n_done} tiles, skipped {n_skip} (no labels) -> {root}/")
    print(f"[grove] under-canopy road = {ucr_px} px ({pct:.1f}% of {road_px} road px)"
          + (f" | oriented road px = {orient_px}" if do_orient else ""))
    print("[grove] added per-tile keys: under_canopy_road [H,W], "
          + ("orient [2,H,W] (sin2θ,cos2θ)" if do_orient else "(orientation disabled)"))
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description="GROVE Stage 1: under-canopy + orientation targets")
    ap.add_argument("--config", default="config/phase1/grove.yaml")
    args = ap.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
