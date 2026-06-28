"""Phase 2 orchestrator — road-mask GeoTIFF -> healed routable graph + artifacts.

  read_mask -> [tiled | window | full] build -> georeference
            -> heal (Union-Find + MST) -> weight -> metrics -> export

Everything is config-driven (config/phase2/config_phase2.yaml -> graph):
  mask          : input GeoTIFF (pred_mask.tif or OSM mask). null => build OSM (make_mask).
  tiling.enabled: true  => process the whole scene in blocks (sknw can't do it whole).
  window        : [row,col,h,w] => process ONE sub-region (only when tiling.enabled: false).
  (else)        : full-scene single pass (only for small masks).

Outputs: runs/graph/<ts>/ {graph.graphml (Phase 3 input), roads.geojson, metrics.csv,
figures/graph_overlay (single/window mode)}.

RUN:  python -m src.phase2.graph.run_graph --config config/phase2/config_phase2.yaml
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

from ...common.config import load_config
from . import (binarize, build, georef, heal, io, make_osm_mask, metrics,
               skeleton, tile, weights)


def run(cfg: dict) -> Path:
    g = cfg.get("graph", {})
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(g.get("out_dir", "runs/graph")) / stamp
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # --- input mask: provided GeoTIFF, or build an OSM mask (dev) --------------
    mask_path = g.get("mask")
    if not mask_path:
        mm = g.get("make_mask") or {}
        if not mm.get("ref_raster"):
            raise RuntimeError("Set graph.mask (a GeoTIFF) OR graph.make_mask.ref_raster.")
        print("[graph] no mask given -> building OSM mask (dev) ...")
        mask_path = make_osm_mask.build_osm_mask(
            mm["ref_raster"], mm.get("aoi"), str(out / "osm_mask.tif"),
            mm.get("network_type", "drive"), float(mm.get("buffer_m", 4.0)))

    print(f"[graph] 1/8 reading mask {mask_path}")
    arr, transform, crs = io.read_mask(mask_path)

    tiling_on = bool(g.get("tiling", {}).get("enabled"))
    skel = None  # set only in single/window mode (for the overlay)

    if tiling_on:
        bs = int(g.get("tiling", {}).get("block_size", 2048))
        print(f"[graph] 2-4/8 TILED build (block={bs}px) over {arr.shape[0]}x{arr.shape[1]} ...")
        G, nblk, nne = tile.build_graph_tiled(arr, transform, crs, g)
        print(f"[graph]     {nne}/{nblk} non-empty blocks")
    else:
        win = g.get("window")
        if win:
            from rasterio.windows import Window, transform as _win_tf
            r0, c0, h0, w0 = (int(x) for x in win)
            arr = arr[r0:r0 + h0, c0:c0 + w0]
            transform = _win_tf(Window(c0, r0, w0, h0), transform)
            print(f"[graph]     window crop row={r0} col={c0} -> {arr.shape[0]}x{arr.shape[1]}")
        print(f"[graph] 2/8 binarize+clean (thr={g.get('threshold',0.5)})")
        binary = binarize.clean_binary(arr, g.get("threshold", 0.5),
                                       g.get("min_object_size", 50), g.get("closing_radius", 2))
        rf = float(binary.mean())
        if not win and rf > float(g.get("max_road_frac", 0.05)):
            print(f"[graph] !! WARNING road fraction {rf:.1%} on a FULL scene — sknw may "
                  "segfault. Use tiling.enabled: true, or a window, or a better mask.")
        print(f"[graph] 3/8 skeletonize ({int(binary.sum())} road px)")
        skel = skeleton.skeletonize_mask(binary)
        print("[graph] 4/8 build graph (sknw)")
        G = build.build_graph(skel)
        georef.georeference(G, transform, crs)

    before = metrics.graph_stats(G)
    G_before = G.copy() if skel is not None else None
    print(f"[graph]     graph: {before['n_nodes']} nodes, {before['n_edges']} edges, "
          f"{before['n_components']} components (LCC={before['largest_cc_nodes']})")

    hc = g.get("heal", {})
    print(f"[graph] 6/8 heal (gap<={hc.get('max_gap_m',60)}m, angle<={hc.get('max_angle_deg',30)}deg)")
    G, n_bridges = heal.heal_graph(G, hc.get("max_gap_m", 60.0),
                                   hc.get("max_angle_deg", 30.0), hc.get("angle_penalty", 1.0))
    after = metrics.graph_stats(G)
    cr = metrics.connectivity_ratio(before["largest_cc_nodes"], after["largest_cc_nodes"])
    print(f"[graph]     healed {n_bridges} bridges | components {before['n_components']}"
          f"->{after['n_components']} | Connectivity Ratio = {cr:.3f}")

    print(f"[graph] 7/8 weights (speed={g.get('speed_kph_default',30)} kph)")
    weights.add_weights(G, g.get("speed_kph_default", 30.0))

    print("[graph] 8/8 export")
    io.write_graph(G, str(out / "graph.graphml"))
    io.write_geojson(G, crs, str(out / "roads.geojson"))
    if G_before is not None and skel is not None:
        try:
            io.save_overlay(skel, G_before, G, out / "figures", "graph_overlay")
        except Exception as e:  # noqa: BLE001
            print(f"[graph] overlay skipped ({e})")
    else:
        print("[graph] overlay skipped (tiled mode) — view roads.geojson in QGIS")

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before", "after"])
        for k in ("n_nodes", "n_edges", "n_components", "largest_cc_nodes", "total_length_km"):
            w.writerow([k, before[k], after[k]])
        w.writerow(["healed_bridges", "", n_bridges])
        w.writerow(["connectivity_ratio", "", f"{cr:.4f}"])

    print(f"[graph] DONE -> {out}  (graph.graphml = Phase 3 input)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 — road mask -> healed graph (config-driven)")
    ap.add_argument("--config", default="config/phase2/config_phase2.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
