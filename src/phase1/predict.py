"""Phase 1 -> Phase 2 export: trained model -> georeferenced pred_mask.tif.

Loads a checkpoint, runs windowed inference over the whole LISS-IV scene, and writes
a single-band georeferenced GeoTIFF of road probability (the Phase 2 contract). Reuses
the ingest's memory-safe windowed readers (_Aligned / WarpedVRT).

Stack = [G, R, NIR, NDVI] (PS-minimal 4-ch). Normalisation matches training
(cfg.data.norm). Non-overlapping tiles; the <tile_size border strip is left 0.

RUN:  python -m src.phase1.predict --ckpt runs/train/<ts>/best.pt --out data/pred_mask.tif
(model architecture + norm are read from the checkpoint's saved cfg.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..common.config import load_config
from ..common.runtime import describe_runtime, get_device
from .data.indices import ndvi as compute_ndvi
from .models import build_model
from .preprocess.ingest_liss4 import _Aligned, _resolve_band_paths, _pbar


def predict(cfg: dict, ckpt_path: str, out_path: str, binary: bool = False) -> str:
    import rasterio
    from rasterio.windows import Window

    device = get_device(cfg.get("runtime", {}).get("device", "auto"))
    print(f"[predict] {describe_runtime(device)}")

    model = build_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    print(f"[predict] loaded {ckpt_path}")

    lc = cfg["data"]["liss4"]
    ts = int(lc.get("tile_size", cfg["data"].get("tile_size", 256)))
    stack, gp, rp, npth = _resolve_band_paths(lc)
    ref_path = stack if stack else gp
    with rasterio.open(ref_path) as ref:
        ref_crs, ref_tf, W, H = ref.crs, ref.transform, ref.width, ref.height
        nodata = lc.get("nodata", ref.nodata)

    if stack:
        sds = rasterio.open(stack); readers = None
    else:
        readers = {"g": _Aligned(gp, ref_crs, ref_tf, W, H),
                   "r": _Aligned(rp, ref_crs, ref_tf, W, H),
                   "n": _Aligned(npth, ref_crs, ref_tf, W, H)}
        sds = None

    nz = cfg["data"].get("norm") or {}
    mean = np.asarray(nz["mean"], "float32") if nz.get("mean") else None
    std = np.asarray(nz["std"], "float32") if nz.get("std") else None
    print(f"[predict] normalization: {'ON' if mean is not None else 'OFF'} | tiles {ts}px")

    out_dtype = "uint8" if binary else "float32"
    thr = float(cfg.get("eval", {}).get("threshold", 0.5))
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype=out_dtype,
                crs=ref_crs, transform=ref_tf, compress="deflate", nodata=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    coords = [(r, c) for r in range(0, H - ts + 1, ts) for c in range(0, W - ts + 1, ts)]
    with rasterio.open(out_path, "w", **prof) as dst, torch.no_grad():
        for (r, c) in _pbar(coords, desc="predict", unit="tile"):
            win = Window(c, r, ts, ts)
            if stack:
                g = sds.read(1, window=win).astype("float32")
                rd = sds.read(2, window=win).astype("float32")
                ni = sds.read(3, window=win).astype("float32")
            else:
                g = readers["g"].read_window(win)
                rd = readers["r"].read_window(win)
                ni = readers["n"].read_window(win)
            valid = (g != nodata) if nodata is not None else (g != 0)
            if not valid.any():
                continue
            nd = compute_ndvi(ni, rd).astype("float32")
            x = np.stack([g, rd, ni, nd]).astype("float32")           # [4,ts,ts]
            if mean is not None and std is not None:
                x = (x - mean[:, None, None]) / (std[:, None, None] + 1e-6)
            t = torch.from_numpy(x[None]).to(device)                  # [1,4,ts,ts]
            prob = torch.sigmoid(model(t)).squeeze().float().cpu().numpy()
            prob = prob * valid                                       # zero nodata
            dst.write((prob >= thr).astype("uint8") if binary else prob.astype("float32"),
                      1, window=win)

    if stack:
        sds.close()
    else:
        for rd_ in readers.values():
            rd_.close()
    print(f"[predict] DONE -> {out_path}  (the Phase 2 input: graph.mask)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 -> pred_mask.tif export")
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--config", default="config/phase1/config.yaml",
                    help="fallback config if the checkpoint has none")
    ap.add_argument("--out", default="data/pred_mask.tif")
    ap.add_argument("--binary", action="store_true", help="write 0/1 instead of probability")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg") or load_config(args.config)   # checkpoint cfg = training settings
    predict(cfg, args.ckpt, args.out, binary=args.binary)


if __name__ == "__main__":
    main()
