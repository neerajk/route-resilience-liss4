"""Phase I training entrypoint — device-dynamic (MPS / CUDA / CPU).

Run (dep-free smoke test on M1):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config config/config.yaml

What it does
------------
1. Resolve device (CUDA->MPS->CPU) and seed everything (reproducibility).
2. Build train/val Datasets + DataLoaders from config (synthetic | real tiles).
3. Build the model (miniunet | smp | dinov3 | clay) via the factory.
4. Optimise CombinedRoadLoss (BCE + Dice + clDice, optional canopy-weighting).
5. Each epoch: log component losses; on val compute the metric suite with
   GLOBALLY-POOLED IoU/Dice/Occlusion-Recall (unbiased) + buffered means.
6. Save to runs/train/<timestamp>/: best checkpoint, metrics.csv, loss curve,
   and a qualitative prediction panel. These ARE the paper artifacts.

NOTE: AMP autocast + GradScaler are CUDA-only; on MPS/CPU we run fp32 (no-op).
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.augment import build_augment
from .data.dataset import DEFAULT_CHANNELS, SyntheticRoadDataset, TileFolderDataset
from .losses import CombinedRoadLoss
from .metrics import pixel_counts, relaxed_iou, relaxed_prf
from .models import build_model
from .utils import amp_autocast, describe_runtime, get_device, set_seed
from .viz.plots import save_fig, save_prediction_panel, set_pub_style


def _pbar(iterable, **kw):
    """tqdm progress bar if available, else the plain iterable (graceful)."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except ImportError:
        return iterable


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` onto `base` (override wins; dicts merged)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_config(path: str) -> dict:
    """Load YAML config. Supports `extends: <sibling.yaml>` to inherit a base
    config and override only a few keys (e.g. config_gpu.yaml extends config.yaml)."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        base_path = Path(path).parent / base
        with open(base_path) as bf:
            cfg = _deep_merge(yaml.safe_load(bf) or {}, cfg)
    return cfg


def _parse_norm(cfg: dict):
    """Per-channel (mean,std) from cfg.data.norm, or None (no standardisation)."""
    nz = cfg.get("data", {}).get("norm", {}) or {}
    mean, std = nz.get("mean"), nz.get("std")
    if mean is None or std is None:
        return None
    return np.asarray(mean, "float32"), np.asarray(std, "float32")


def _worker_init(worker_id: int) -> None:
    """Reproducible, worker-INDEPENDENT augmentation RNG (avoids identical seeds)."""
    base = int(torch.initial_seed()) % (2 ** 32)
    np.random.seed((base + worker_id) % (2 ** 32))


def _datasets(cfg: dict, norm):
    d = cfg["data"]
    channels = tuple(d.get("channels", DEFAULT_CHANNELS))
    size = int(d.get("tile_size", 256))
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    syn = d.get("synthetic", {})
    aug = build_augment(cfg)   # train-time only; None unless augment.enabled
    if d.get("source", "synthetic") == "synthetic":
        common = dict(size=size, channels=channels,
                      canopy_fraction=float(syn.get("canopy_fraction", 0.35)),
                      n_roads=int(syn.get("n_roads", 9)),
                      road_width=int(syn.get("road_width", 1)), norm=norm)
        train = SyntheticRoadDataset(d.get("train_samples", 64), seed=seed, augment=aug, **common)
        val = SyntheticRoadDataset(d.get("val_samples", 16), seed=seed + 10_000,
                                   augment=None, **common)   # never augment validation
        return train, val
    # ===== real LISS-IV tiles (see TileFolderDataset USER INPUT) =====
    # NOTE: full spatial-block CV (cfg.data.cv) is applied at tiling time
    # (preprocess); here we use a simple contiguous val split as a fallback.
    full = TileFolderDataset(root=d["root"], channels=channels, augment=aug, norm=norm)
    vf = float(d.get("cv", {}).get("val_fraction", 0.2))
    n_val = max(1, int(vf * len(full)))
    val = torch.utils.data.Subset(full, range(n_val))
    train = torch.utils.data.Subset(full, range(n_val, len(full)))
    return train, val


@torch.no_grad()
def _validate(model, loader, device, buffer: int, thr: float):
    """Headline IoU/Dice/Occlusion-Recall pooled over GLOBAL pixel counts
    (unbiased); buffered precision/recall/F1/IoU as per-batch means."""
    model.eval()
    tot = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "occ_tp": 0.0, "occ_total": 0.0}
    rel = {"relaxed_precision": 0.0, "relaxed_recall": 0.0, "relaxed_f1": 0.0, "relaxed_iou": 0.0}
    n = 0
    for batch in _pbar(loader, desc="  validating", unit="batch", leave=False):
        logits = model(batch["image"].to(device)).float().cpu()
        y, c = batch["mask"], batch["canopy"]
        pc = pixel_counts(logits, y, c, thr=thr)
        for k in tot:
            tot[k] += pc[k]
        r = relaxed_prf(logits, y, buffer=buffer, thr=thr)
        for k in ("relaxed_precision", "relaxed_recall", "relaxed_f1"):
            rel[k] += r[k]
        rel["relaxed_iou"] += relaxed_iou(logits, y, buffer=buffer, thr=thr)
        n += 1
    eps = 1e-6
    out = {
        "iou": (tot["tp"] + eps) / (tot["tp"] + tot["fp"] + tot["fn"] + eps),
        "dice": (2 * tot["tp"] + eps) / (2 * tot["tp"] + tot["fp"] + tot["fn"] + eps),
        "occlusion_recall": (tot["occ_tp"] + eps) / (tot["occ_total"] + eps),
    }
    for k, v in rel.items():
        out[k] = v / max(n, 1)
    return out


def run(cfg: dict) -> Path:
    rt = cfg.get("runtime", {})
    set_seed(int(rt.get("seed", 42)))
    device = get_device(rt.get("device", "auto"))
    print(f"[train] {describe_runtime(device)}")
    _d, _m = cfg["data"], cfg["model"]
    print(f"[train] source={_d.get('source')} | channels={_d.get('channels')} | "
          f"arch={_m.get('arch')} encoder={_m.get('encoder')} decoder={_m.get('decoder')} "
          f"in_ch={_m.get('in_channels')}")

    norm = _parse_norm(cfg)
    print(f"[train] normalization: {'ON' if norm is not None else 'OFF (raw inputs)'}")
    train_ds, val_ds = _datasets(cfg, norm)
    tr = cfg["train"]
    nw = int(tr.get("num_workers", 0))
    wif = _worker_init if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=int(tr.get("batch_size", 4)),
                              shuffle=True, num_workers=nw, drop_last=True, worker_init_fn=wif)
    val_loader = DataLoader(val_ds, batch_size=int(tr.get("batch_size", 4)),
                            shuffle=False, num_workers=nw)

    print(f"[train] building model '{_m.get('arch')}' ...")
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] datasets: train={len(train_ds)} val={len(val_ds)} | "
          f"batch={int(tr.get('batch_size', 4))} workers={nw} | "
          f"trainable params={n_par / 1e6:.2f}M")
    lcfg = cfg.get("loss", {})
    w = lcfg.get("weights", {})
    criterion = CombinedRoadLoss(
        w_bce=w.get("bce", 0.3), w_dice=w.get("dice", 0.4), w_cldice=w.get("cldice", 0.3),
        cldice_iters=int(lcfg.get("cldice_iters", 10)),
        canopy_weight=float(lcfg.get("canopy_weight", 0.0)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.get("lr", 1e-3)),
                            weight_decay=float(tr.get("weight_decay", 1e-4)))
    use_amp = bool(tr.get("amp", True))
    cuda_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=cuda_amp)   # no-op off CUDA
    buffer = int(cfg.get("eval", {}).get("relax_buffer_px", 3))
    thr = float(cfg.get("eval", {}).get("threshold", 0.5))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg["paths"]["runs"]) / "train" / stamp
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = cfg.get("eval", {}).get("monitor", "occlusion_recall")
    best_val = -1.0

    epochs = int(tr.get("epochs", 3))
    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = {"total": 0.0, "bce": 0.0, "dice": 0.0, "cldice": 0.0}
        steps = 0
        bar = _pbar(train_loader, desc=f"epoch {ep}/{epochs}", unit="batch", leave=False)
        for batch in bar:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            c = batch["canopy"].to(device)
            opt.zero_grad()
            with amp_autocast(device, enabled=use_amp):
                logits = model(x)
                loss, comps = criterion(logits, y, c)   # canopy-weighted if configured
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_losses["total"] += loss.item()
            for k in ("bce", "dice", "cldice"):
                ep_losses[k] += comps[k]
            steps += 1
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss.item():.3f}", cldice=f"{comps['cldice']:.3f}")
        ep_losses = {k: v / max(steps, 1) for k, v in ep_losses.items()}

        val_metrics = _validate(model, val_loader, device, buffer, thr)
        row = {"epoch": ep, **{f"loss_{k}": v for k, v in ep_losses.items()}, **val_metrics}
        history.append(row)
        print(f"[ep {ep}/{epochs}] loss={ep_losses['total']:.4f} "
              f"IoU={val_metrics['iou']:.3f} OccRec={val_metrics['occlusion_recall']:.3f} "
              f"relaxedF1={val_metrics['relaxed_f1']:.3f}")

        if val_metrics.get(best_key, -1) > best_val:
            best_val = val_metrics[best_key]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                        "val": val_metrics}, out / "best.pt")

    pd.DataFrame(history).to_csv(out / "metrics.csv", index=False)

    # ---- figures: loss curve + qualitative prediction panel (paper artifacts) ----
    set_pub_style()
    import matplotlib.pyplot as plt
    hdf = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(hdf["epoch"], hdf["loss_total"], marker="o", label="total")
    ax.plot(hdf["epoch"], hdf["loss_cldice"], marker="s", label="clDice")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss")
    ax.legend()
    save_fig(fig, fig_dir, "loss_curve")

    try:
        model.eval()
        vb = next(iter(val_loader))
        with torch.no_grad():
            vlog = model(vb["image"].to(device)).float().cpu()
        save_prediction_panel(vb["image"][0], vb["mask"][0], vb["canopy"][0],
                              vlog[0], fig_dir, "prediction", thr=thr)
    except Exception as e:  # noqa: BLE001 - artifact is best-effort
        print(f"[train] prediction panel skipped ({e})")

    print(f"[train] best {best_key}={best_val:.3f} | artifacts -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase I training")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    run(_load_config(args.config))


if __name__ == "__main__":
    main()
