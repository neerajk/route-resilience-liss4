"""GROVE multi-task training entrypoint (Stages 3-4): seg + orientation heads.

Compact trainer for the FULL GROVE arm (orientation head + under-canopy focal +
clDice). For the seg-only BACKBONE BENCHMARK (Stage 2), use the existing VISTA
trainer instead — a seg-only GroveNet returns plain logits, so:
    python -m src.phase1.vista.train --config config/phase1/grove_<backbone>.yaml
gets you the full CV/TTA pipeline. This trainer adds the orientation supervision.

RUN:
    python -m src.phase1.grove.train --config config/phase1/grove.yaml

Device-dynamic (MPS/CUDA/CPU). Artifacts → runs/train/grove__<backbone>__<stage>__<ts>/.
NOTE: run with geometric augmentation OFF (orientation-aware aug not yet wired).
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import GroveTileDataset
from .losses import GroveLoss
from ..shared.data.dataset import DEFAULT_CHANNELS, SyntheticRoadDataset
from ..shared.metrics import pixel_counts
from ..shared.models import build_model
from ...common.config import load_config
from ...common.naming import run_dir
from ...common.runtime import describe_runtime, get_device, set_seed


def _pbar(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


def _norm(cfg):
    nz = cfg.get("data", {}).get("norm", {}) or {}
    if nz.get("mean") and nz.get("std"):
        return (np.asarray(nz["mean"], "float32"), np.asarray(nz["std"], "float32"))
    return None


def _datasets(cfg):
    d = cfg.get("data", {})
    channels = tuple(d.get("channels", DEFAULT_CHANNELS))
    if str(d.get("source", "synthetic")) == "tiles":
        ds = GroveTileDataset(root=d.get("root", "data/tiles"), channels=channels,
                              augment=None, norm=_norm(cfg))
        n_val = max(1, int(len(ds) * float(d.get("cv", {}).get("val_fraction", 0.2))))
        val = torch.utils.data.Subset(ds, list(range(n_val)))
        train = torch.utils.data.Subset(ds, list(range(n_val, len(ds))))
        return train, val
    # synthetic fallback (smoke): no orient key -> orientation term auto-skips
    sy = cfg.get("data", {}).get("synthetic", {})
    common = dict(size=int(d.get("tile_size", 256)), channels=channels,
                  canopy_fraction=float(sy.get("canopy_fraction", 0.35)),
                  n_roads=int(sy.get("n_roads", 9)), road_width=int(sy.get("road_width", 1)))
    train = SyntheticRoadDataset(length=int(d.get("train_samples", 64)), seed=0, **common)
    val = SyntheticRoadDataset(length=int(d.get("val_samples", 16)), seed=10_000, **common)
    return train, val


def run(cfg: dict):
    device = get_device(cfg.get("runtime", {}).get("device", "auto"))
    set_seed(int(cfg.get("runtime", {}).get("seed", 42)))
    print(f"[grove-train] {describe_runtime(device)}")

    train_ds, val_ds = _datasets(cfg)
    t = cfg.get("train", {})
    bs = int(t.get("batch_size", 4))
    nw = int(t.get("num_workers", 0))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = build_model(cfg).to(device)
    lw = cfg.get("loss", {}).get("weights", {})
    loss_fn = GroveLoss(
        w_bce=float(lw.get("bce", 0.3)), w_dice=float(lw.get("dice", 0.4)),
        w_cldice=float(lw.get("cldice", 0.3)),
        cldice_iters=int(cfg.get("loss", {}).get("cldice_iters", 6)),
        canopy_weight=float(cfg.get("loss", {}).get("canopy_weight", 1.5)),
        ucr_weight=float(cfg.get("loss", {}).get("ucr_weight", 2.0)),
        orient_weight=float(cfg.get("loss", {}).get("orient_weight", 0.5)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 1e-3)),
                            weight_decay=float(t.get("weight_decay", 1e-4)))
    epochs = int(t.get("epochs", 50))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=float(t.get("lr", 1e-3)) * 1e-3)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = run_dir(cfg, stamp, "train")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[grove-train] -> {out}")

    best, rows = -1.0, []
    for ep in range(epochs):
        model.train()
        for batch in _pbar(train_dl, desc=f"ep{ep}", unit="b"):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            canopy = batch["canopy"].to(device)
            ucr = mask * canopy                                  # under-canopy, in sync
            orient = batch["orient"].to(device) if "orient" in batch else None
            opt.zero_grad()
            out_pred = model(img)
            loss, comp = loss_fn(out_pred, mask, canopy=canopy, under_canopy=ucr,
                                 orient_target=orient)
            loss.backward()
            opt.step()
        sched.step()

        # validation — global pixel pooling (unbiased), seg head only
        model.eval()
        agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "occ_tp": 0.0, "occ_total": 0.0}
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                canopy = batch["canopy"].to(device)
                pred = model(img)
                seg = pred["seg"] if isinstance(pred, dict) else pred
                pc = pixel_counts(seg, mask, canopy, logits=True,
                                  thr=float(cfg.get("eval", {}).get("threshold", 0.5)))
                for k in agg:
                    agg[k] += pc[k]
        eps = 1e-6
        iou = agg["tp"] / (agg["tp"] + agg["fp"] + agg["fn"] + eps)
        dice = 2 * agg["tp"] / (2 * agg["tp"] + agg["fp"] + agg["fn"] + eps)
        occ_rec = agg["occ_tp"] / (agg["occ_total"] + eps)
        rows.append({"epoch": ep, "iou": iou, "dice": dice, "occlusion_recall": occ_rec,
                     **{f"loss_{k}": v for k, v in comp.items()}})
        print(f"[grove-train] ep{ep}: OccRec={occ_rec:.4f} IoU={iou:.4f} Dice={dice:.4f}")
        if occ_rec > best:
            best = occ_rec
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                        "occlusion_recall": occ_rec}, out / "best.pt")
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(f"[grove-train] DONE best OccRec={best:.4f} -> {out}/best.pt")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GROVE multi-task training (seg + orientation)")
    ap.add_argument("--config", default="config/phase1/grove.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
