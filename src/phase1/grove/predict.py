"""GROVE export — trained GroveNet → georeferenced mask (+ orientation) for Phase 2.

Windowed inference over the whole LISS-IV scene (reuses the ingest's memory-safe
readers). Writes:
  data/grove__pred_mask.tif    1-band road probability (Phase-2 contract)
  data/grove__orientation.tif  2-band (sin2θ,cos2θ) — only if the model has the
                               orientation head; feeds Phase 2 heal.mode: orientation.

RUN:  python -m src.phase1.grove.predict --ckpt runs/train/grove__.../best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..shared.data.indices import ndvi as compute_ndvi
from ..shared.models import build_model
from ..shared.preprocess.ingest_liss4 import _Aligned, _pbar, _resolve_band_paths
from ...common.config import load_config
from ...common.naming import orientation_path, pred_mask_path
from ...common.runtime import describe_runtime, get_device


def predict(cfg: dict, ckpt_path: str, out_mask: str, out_orient: str | None,
            binary: bool = False) -> str:
    import rasterio
    from rasterio.windows import Window

    device = get_device(cfg.get("runtime", {}).get("device", "auto"))
    print(f"[grove-predict] {describe_runtime(device)}")
    model = build_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

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
    thr = float(cfg.get("eval", {}).get("threshold", 0.5))

    base = dict(driver="GTiff", height=H, width=W, crs=ref_crs, transform=ref_tf,
                compress="deflate")
    Path(out_mask).parent.mkdir(parents=True, exist_ok=True)
    mask_prof = dict(base, count=1, dtype="uint8" if binary else "float32", nodata=0)
    dst_or = None
    coords = [(r, c) for r in range(0, H - ts + 1, ts) for c in range(0, W - ts + 1, ts)]

    with rasterio.open(out_mask, "w", **mask_prof) as dst, torch.no_grad():
        for (r, c) in _pbar(coords, desc="grove-predict", unit="tile"):
            win = Window(c, r, ts, ts)
            if stack:
                g = sds.read(1, window=win).astype("float32")
                rd = sds.read(2, window=win).astype("float32")
                ni = sds.read(3, window=win).astype("float32")
            else:
                g = readers["g"].read_window(win); rd = readers["r"].read_window(win)
                ni = readers["n"].read_window(win)
            valid = (g != nodata) if nodata is not None else (g != 0)
            if not valid.any():
                continue
            nd = compute_ndvi(ni, rd).astype("float32")
            x = np.stack([g, rd, ni, nd]).astype("float32")
            if mean is not None and std is not None:
                x = (x - mean[:, None, None]) / (std[:, None, None] + 1e-6)
            out = model(torch.from_numpy(x[None]).to(device))
            seg = out["seg"] if isinstance(out, dict) else out
            prob = (torch.sigmoid(seg).squeeze().float().cpu().numpy()) * valid
            dst.write((prob >= thr).astype("uint8") if binary else prob.astype("float32"),
                      1, window=win)
            if isinstance(out, dict) and "orient" in out and out_orient is not None:
                if dst_or is None:
                    dst_or = rasterio.open(out_orient, "w", **dict(base, count=2, dtype="float32",
                                                                  nodata=0))
                ori = out["orient"].squeeze(0).float().cpu().numpy() * valid
                dst_or.write(ori[0].astype("float32"), 1, window=win)
                dst_or.write(ori[1].astype("float32"), 2, window=win)

    if dst_or is not None:
        dst_or.close()
        print(f"[grove-predict] orientation -> {out_orient}")
    if stack:
        sds.close()
    else:
        for rd_ in readers.values():
            rd_.close()
    print(f"[grove-predict] DONE -> {out_mask}")
    return out_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="GROVE export -> grove__pred_mask.tif (+orientation)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config/phase1/grove.yaml")
    ap.add_argument("--out", default=None, help="default data/<arm>__pred_mask.tif")
    ap.add_argument("--binary", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg") or load_config(args.config)
    out_mask = args.out or str(pred_mask_path(cfg, "data", binary=args.binary))
    out_or = str(orientation_path(cfg, "data"))
    predict(cfg, args.ckpt, out_mask, out_or, binary=args.binary)


if __name__ == "__main__":
    main()
