"""Stage-1 ingestion: LISS-IV band GeoTIFFs -> model-ready .npz tiles (+ OSM labels).

USE CASE: you provide LISS-IV B2/B3/B4 = Green/Red/NIR GeoTIFFs (three files OR one
3-band stack). This:
  1. Sets the reference grid = the Green band's CRS/transform/shape.
  2. Reads Red/NIR per-tile (WarpedVRT-aligned if their grid differs — memory-safe).
  3. Computes NDVI = (NIR-Red)/(NIR+Red) per tile.
  4. **Labels (Step 1):** auto-pulls OSM roads (osmnx) for the AOI and rasterises
     them onto the LISS-IV grid -> per-tile road `mask` (zero manual labelling).
  5. Occlusion proxy `canopy` for Occlusion-Recall: NDVI > thresh (PS-minimal,
     no CHM). If an optional CHM GeoTIFF is given, canopy = CHM-tall AND NDVI-veg.
  6. Skips tiles with too few VALID pixels; writes band statistics for normalisation.

OUTPUT tile schema (.npz): bands[3,ts,ts] (G,R,NIR), ndvi[ts,ts], canopy[ts,ts],
  mask[ts,ts] (if labels=osm), [chm[ts,ts] if CHM provided], row, col, bounds.

RUN:  python -m src.phase1.shared.preprocess.ingest_liss4 --config config/phase1/config.yaml
Config: cfg.data.liss4 (paths, aoi, labels, buffer_m, tiling).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pbar(iterable, **kw):
    """tqdm progress bar if available, else the plain iterable (graceful)."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except ImportError:
        return iterable


def _grid_matches(ds, ref_crs, ref_transform, ref_w: int, ref_h: int) -> bool:
    return (ds.crs == ref_crs and ds.transform == ref_transform
            and ds.width == ref_w and ds.height == ref_h)


class _Aligned:
    """Read band-1 windows from a raster, resampled onto a reference grid if needed."""

    def __init__(self, path: str, ref_crs, ref_transform, ref_w: int, ref_h: int,
                 resampling: str = "bilinear") -> None:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        self._ds = rasterio.open(path)
        if _grid_matches(self._ds, ref_crs, ref_transform, ref_w, ref_h):
            self._src, self._vrt = self._ds, None
        else:
            rs = {"bilinear": Resampling.bilinear, "cubic": Resampling.cubic,
                  "nearest": Resampling.nearest}[resampling]
            self._vrt = WarpedVRT(self._ds, crs=ref_crs, transform=ref_transform,
                                  width=ref_w, height=ref_h, resampling=rs)
            self._src = self._vrt

    def read_window(self, window) -> np.ndarray:
        return self._src.read(1, window=window).astype("float32")

    def close(self) -> None:
        if self._vrt is not None:
            self._vrt.close()
        self._ds.close()


def _resolve_band_paths(lc: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (stack | None, green, red, nir). One of stack OR the trio."""
    if lc.get("stack"):
        return lc["stack"], None, None, None
    g, r, n = lc.get("green"), lc.get("red"), lc.get("nir")
    if not (g and r and n):
        raise RuntimeError("Set cfg.data.liss4: `stack` OR `green`/`red`/`nir` paths.")
    return None, g, r, n


def _aoi_bbox(aoi_path, ref_crs, ref_tf, H: int, W: int) -> List[float]:
    """[minlon,minlat,maxlon,maxlat] in EPSG:4326 from the AOI shapefile, else the
    image bounds. osmnx needs lon/lat."""
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    if aoi_path:
        import geopandas as gpd
        g = gpd.read_file(aoi_path).to_crs("EPSG:4326")
        return [float(x) for x in g.total_bounds]
    w, s, e, n = array_bounds(H, W, ref_tf)
    return list(transform_bounds(ref_crs, "EPSG:4326", w, s, e, n))


def _build_osm_mask(bbox, ref_crs, ref_tf, H: int, W: int,
                    network_type: str, buffer_m: float) -> Optional[np.ndarray]:
    """Fetch OSM roads for bbox and rasterise to a full-scene [H,W] uint8 mask."""
    from ..data.sources import osm
    print(f"[ingest] fetching OSM '{network_type}' roads for bbox {[round(x,4) for x in bbox]} ...")
    try:
        _, edges = osm.fetch_osm_roads(bbox, network_type=network_type)
    except Exception as e:  # noqa: BLE001 - degrade to imagery-only
        print(f"[ingest] OSM fetch failed ({e}); writing tiles WITHOUT mask.")
        return None
    mask = osm.rasterize_roads(edges, ref_tf, (H, W), crs=str(ref_crs), buffer_m=buffer_m)
    print(f"[ingest] OSM mask: {int(mask.sum())} road px ({100 * mask.mean():.3f}% of scene)")
    return mask


def ingest(cfg: dict) -> Path:
    import rasterio
    from rasterio.windows import Window, bounds as win_bounds

    from ..data.indices import ndvi as compute_ndvi

    lc = cfg.get("data", {}).get("liss4", {})
    if not lc:
        raise RuntimeError("cfg.data.liss4 block missing — see config.yaml.")
    ts = int(lc.get("tile_size", cfg.get("data", {}).get("tile_size", 256)))
    stride = int(lc.get("stride", ts))
    mvf = float(lc.get("min_valid_frac", 0.5))
    maxt = int(lc.get("max_tiles", 0))   # 0 = all (use a small value for quick tests)
    ndvi_thresh = float(lc.get("ndvi_thresh", 0.3))
    chm_thresh = float(lc.get("canopy_height_thresh", 3.0))
    out_dir = Path(lc.get("out_dir", "data/tiles"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stack, gp, rp, npth = _resolve_band_paths(lc)
    ref_path = stack if stack else gp
    with rasterio.open(ref_path) as ref:
        ref_crs, ref_tf, W, H = ref.crs, ref.transform, ref.width, ref.height
        nodata = lc.get("nodata", ref.nodata)
        gsd = abs(ref_tf.a)
    print(f"[ingest] reference grid {W}x{H}  CRS={ref_crs}  GSD~{gsd:.2f}  nodata={nodata}")

    # band readers
    if stack:
        green_ds, readers = rasterio.open(stack), None
    else:
        readers = {"green": _Aligned(gp, ref_crs, ref_tf, W, H),
                   "red": _Aligned(rp, ref_crs, ref_tf, W, H),
                   "nir": _Aligned(npth, ref_crs, ref_tf, W, H)}
        green_ds = None
    chm_path = lc.get("chm")
    chm_reader = _Aligned(chm_path, ref_crs, ref_tf, W, H, "bilinear") if chm_path else None

    # labels (OSM) -> full-scene mask, windowed per tile
    osm_mask = None
    if str(lc.get("labels", "none")).lower() == "osm":
        bbox = _aoi_bbox(lc.get("aoi"), ref_crs, ref_tf, H, W)
        osm_mask = _build_osm_mask(bbox, ref_crs, ref_tf, H, W,
                                   str(lc.get("network_type", "drive")),
                                   float(lc.get("buffer_m", 4.0)))

    names = ["green", "red", "nir", "ndvi"] + (["chm"] if chm_reader else [])
    s1 = {k: 0.0 for k in names}; s2 = {k: 0.0 for k in names}; cnt = {k: 0 for k in names}

    def _accum(name, arr, valid):
        v = arr[valid].astype("float64")
        s1[name] += float(v.sum()); s2[name] += float((v * v).sum()); cnt[name] += int(v.size)

    coords = [(r, c) for r in range(0, H - ts + 1, stride) for c in range(0, W - ts + 1, stride)]
    print(f"[ingest] {len(coords)} candidate tile positions (size={ts} stride={stride}) | "
          f"labels={'OSM' if osm_mask is not None else 'none'} | chm={'yes' if chm_reader else 'no (NDVI proxy)'}")
    kept = skipped = 0
    bar = _pbar(coords, desc="tiling", unit="tile")
    for (r, c) in bar:
        win = Window(c, r, ts, ts)
        if stack:
            g = green_ds.read(1, window=win).astype("float32")
            red = green_ds.read(2, window=win).astype("float32")
            nir = green_ds.read(3, window=win).astype("float32")
        else:
            g = readers["green"].read_window(win)
            red = readers["red"].read_window(win)
            nir = readers["nir"].read_window(win)

        valid = (g != nodata) if nodata is not None else (g != 0)
        if valid.mean() < mvf:
            skipped += 1
            continue

        nd = compute_ndvi(nir, red).astype("float32")
        if chm_reader is not None:
            chm = chm_reader.read_window(win)
            canopy = ((chm > chm_thresh) & (nd > ndvi_thresh)).astype("float32")
        else:
            chm = None
            canopy = (nd > ndvi_thresh).astype("float32")   # NDVI occlusion proxy

        arrs = [g, red, nir, nd] + ([chm] if chm is not None else [])
        for name, arr in zip(names, arrs):
            _accum(name, arr, valid)

        payload = dict(
            bands=np.stack([g, red, nir]).astype("float32"),
            ndvi=nd, canopy=canopy,
            row=np.int32(r), col=np.int32(c),
            bounds=np.array(win_bounds(win, ref_tf), "float64"),
        )
        if chm is not None:
            payload["chm"] = chm.astype("float32")
        if osm_mask is not None:
            payload["mask"] = osm_mask[r:r + ts, c:c + ts].astype("float32")
        np.savez_compressed(out_dir / f"tile_{kept:05d}.npz", **payload)
        kept += 1
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(kept=kept, skipped=skipped)
        if maxt and kept >= maxt:
            print(f"[ingest] reached max_tiles={maxt}, stopping early.")
            break

    if stack:
        green_ds.close()
    else:
        for rd in readers.values():
            rd.close()
    if chm_reader is not None:
        chm_reader.close()

    # stats + suggested norm (matches cfg.data.channels order)
    means, stds = {}, {}
    for k in names:
        n = max(cnt[k], 1); mu = s1[k] / n
        means[k], stds[k] = mu, max(s2[k] / n - mu * mu, 0.0) ** 0.5
    stats_csv = Path(lc.get("stats_csv", "data/band_statistics.csv"))
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f); wcsv.writerow(["channel", "mean", "std", "count"])
        for k in names:
            wcsv.writerow([k, f"{means[k]:.6f}", f"{stds[k]:.6f}", cnt[k]])

    has_mask = osm_mask is not None
    print(f"[ingest] DONE: wrote {kept} tiles, skipped {skipped} (low-valid) -> {out_dir}/ "
          f"| mask={'OSM' if has_mask else 'NONE'} | stats -> {stats_csv}")
    chans = cfg.get("data", {}).get("channels", ["green", "red", "nir", "ndvi"])
    print("[ingest] set in config.yaml -> data.source: tiles ; data.norm:")
    print(f"    mean: [{', '.join(f'{means.get(c, 0.0):.3f}' for c in chans)}]")
    print(f"    std:  [{', '.join(f'{stds.get(c, 1.0):.3f}' for c in chans)}]")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest LISS-IV band tifs (+OSM labels) -> .npz tiles")
    ap.add_argument("--config", default="config/phase1/config.yaml")
    args = ap.parse_args()
    ingest(_load_config(args.config))


if __name__ == "__main__":
    main()
