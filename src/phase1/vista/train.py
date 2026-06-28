"""Phase I training entrypoint — device-dynamic (MPS / CUDA / CPU).

Run (dep-free smoke test on M1):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.phase1.vista.train --config config/phase1/config.yaml

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

from ..shared.data.augment import build_augment
from ..shared.data.dataset import DEFAULT_CHANNELS, SyntheticRoadDataset, TileFolderDataset
from ..shared.losses import CombinedRoadLoss
from ..shared.metrics import pixel_counts, relaxed_iou, relaxed_prf
from ..shared.models import build_model
from ...common.config import load_config
from ...common.runtime import amp_autocast, describe_runtime, get_device, set_seed
from ...common.viz import save_fig, save_prediction_panel, set_pub_style


def _pbar(iterable, **kw):
    """tqdm progress bar if available, else the plain iterable (graceful)."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except ImportError:
        return iterable


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
    if d.get("source") == "deepglobe":
        # ===== DeepGlobe pretraining: 0.5 m RGB degraded to ~5.8 m =====
        from ..shared.data.deepglobe import DeepGlobeDataset
        dg = d.get("deepglobe", {})
        common = dict(root=dg["root"], tile_size=size,
                      source_gsd_m=float(dg.get("source_gsd_m", 0.5)),
                      target_gsd_m=float(dg.get("target_gsd_m", 5.8)),
                      sat_suffix=dg.get("sat_suffix", "_sat.jpg"),
                      mask_suffix=dg.get("mask_suffix", "_mask.png"),
                      channels=channels,                          # band order (default G,R,B; LISS-IV-aligned)
                      limit=int(dg.get("limit", 0)), norm=norm)
        train_full = DeepGlobeDataset(augment=aug, **common)
        val_full = DeepGlobeDataset(augment=None, **common)   # never augment validation
        vf = float(d.get("cv", {}).get("val_fraction", 0.1))
        order = list(range(len(train_full)))
        np.random.RandomState(seed).shuffle(order)
        n_val = max(1, int(vf * len(order)))
        va_idx, tr_idx = order[:n_val], order[n_val:]
        print(f"[train] DeepGlobe: {len(train_full)} pairs -> train={len(tr_idx)} val={len(va_idx)}")
        return torch.utils.data.Subset(train_full, tr_idx), torch.utils.data.Subset(val_full, va_idx)
    # ===== real LISS-IV tiles (see TileFolderDataset USER INPUT) =====
    # Two dataset views over the SAME tiles: the train view augments, the val view
    # never does (validation must see clean inputs). CV picks which tiles go where.
    train_full = TileFolderDataset(root=d["root"], channels=channels, augment=aug, norm=norm)
    val_full = TileFolderDataset(root=d["root"], channels=channels, augment=None, norm=norm)
    cv = d.get("cv", {}) or {}
    scheme = str(cv.get("scheme", "random")).lower()
    vf = float(cv.get("val_fraction", 0.2))
    if scheme == "spatial_block":
        tr_idx, va_idx = _spatial_block_split(
            train_full.files, float(cv.get("block_size_m", 1500)), vf, seed)
    else:                                                # random/contiguous fallback
        n_val = max(1, int(vf * len(train_full)))
        va_idx, tr_idx = list(range(n_val)), list(range(n_val, len(train_full)))
        print(f"[train] CV: {scheme} contiguous split -> train={len(tr_idx)} val={len(va_idx)} "
              f"(WARNING: spatially adjacent tiles can leak; prefer cv.scheme=spatial_block)")
    train = torch.utils.data.Subset(train_full, tr_idx)
    val = torch.utils.data.Subset(val_full, va_idx)
    return train, val


def _spatial_block_split(files, block_size_m: float, val_fraction: float, seed: int):
    """Assign WHOLE spatial blocks to train/val (Roberts et al. 2017) to stop the
    spatial-autocorrelation leak a random/contiguous tile split causes.

    Each tile's ``bounds`` (projected metres, written by ingest_liss4) maps to a
    block key ``(floor(minx/B), floor(miny/B))``; blocks are shuffled by ``seed``
    and added to val until ~``val_fraction`` of tiles are covered. Falls back to a
    contiguous split (with a warning) if any tile lacks ``bounds``.
    """
    from collections import defaultdict

    blocks: dict = defaultdict(list)
    for i, f in enumerate(files):
        z = np.load(f)
        if "bounds" not in z.files:
            n_val = max(1, int(val_fraction * len(files)))
            print(f"[train] CV: tile {f.name} has no 'bounds' -> falling back to "
                  f"contiguous split (re-run ingest_liss4 to enable spatial blocking).")
            return list(range(n_val, len(files))), list(range(n_val))
        minx, miny = float(z["bounds"][0]), float(z["bounds"][1])
        blocks[(int(np.floor(minx / block_size_m)), int(np.floor(miny / block_size_m)))].append(i)

    block_keys = sorted(blocks)
    np.random.RandomState(seed).shuffle(block_keys)
    target = val_fraction * len(files)
    val_idx: list = []
    for k in block_keys:
        if val_idx and len(val_idx) >= target:
            break
        val_idx.extend(blocks[k])
    val_set = set(val_idx)
    train_idx = [i for i in range(len(files)) if i not in val_set]
    print(f"[train] CV: spatial_block ({block_size_m:.0f} m) -> {len(blocks)} blocks, "
          f"train={len(train_idx)} val={len(val_idx)} tiles (whole blocks held out)")
    return train_idx, val_idx


def load_pretrained(model, path: str, inflate_stem: bool = True) -> dict:
    """Warm-start: copy compatible weights from ANY checkpoint into ``model``.

    Accepts our own checkpoints (``{"model": state_dict, ...}``), Lightning-style
    (``{"state_dict": ...}``), or a raw ``state_dict``. Loading is NON-STRICT and
    shape-checked: only name+shape matches are copied; every mismatch is reported,
    never fatal — so a partial/foreign checkpoint warm-starts what it can and the
    rest stays at initialisation.

    If ``inflate_stem`` and the sole mismatch on a 4-D conv weight is the input-
    channel count with a 3-channel (RGB) source, the stem is inflated via the I3D
    trick (copy RGB onto the first 3 channels, mean-init the extras, rescale by
    3/in_ch; cf. ``models.factory.inflate_first_conv``). This is what lets a
    DeepGlobe/RGB-pretrained model warm-start a [G,R,NIR,NDVI] model.

    Returns a summary dict {loaded, inflated, mismatched, missing}.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"train.init_from checkpoint not found: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        src = obj["model"]
    elif isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        src = obj["state_dict"]
    elif isinstance(obj, dict):
        src = obj                                   # assume a raw state_dict
    else:
        raise ValueError(f"Unrecognised checkpoint format at {path}: {type(obj)}")
    # strip a DataParallel/wrapper prefix if present
    src = {(k[7:] if k.startswith("module.") else k): v for k, v in src.items()}

    tgt = model.state_dict()
    to_load: dict = {}
    inflated: list = []
    mism: list = []
    for k, v in src.items():
        if k not in tgt or not hasattr(v, "shape"):
            continue
        tv = tgt[k]
        if v.shape == tv.shape:
            to_load[k] = v
        elif (inflate_stem and v.dim() == 4 and tv.dim() == 4
              and v.shape[0] == tv.shape[0] and v.shape[2:] == tv.shape[2:]
              and v.shape[1] == 3 and tv.shape[1] >= 3):
            in_ch = tv.shape[1]
            nw = v.new_zeros(tv.shape)
            nw[:, :3] = v                                   # G,R,NIR <- pretrained RGB
            if in_ch > 3:                                   # extras <- channel mean
                nw[:, 3:] = v.mean(dim=1, keepdim=True).repeat(1, in_ch - 3, 1, 1)
            nw.mul_(3.0 / in_ch)                            # preserve magnitude
            to_load[k] = nw
            inflated.append(k)
        else:
            mism.append((k, tuple(v.shape), tuple(tv.shape)))

    missing = model.load_state_dict(to_load, strict=False).missing_keys
    matched = len(to_load) - len(inflated)
    print(f"[warm-start] {path}")
    print(f"[warm-start]   loaded {matched} tensors verbatim"
          + (f" + inflated {len(inflated)} stem conv(s)" if inflated else "")
          + f" | {len(mism)} shape-mismatch skipped | {len(missing)} model tensors left at init")
    for k in inflated:
        print(f"[warm-start]   inflate {k}: RGB(3) -> {tgt[k].shape[1]}ch")
    for k, s, t in mism[:6]:
        print(f"[warm-start]   skip {k}: ckpt{s} vs model{t}")
    if not to_load:
        print("[warm-start]   WARNING: 0 tensors matched — check the checkpoint matches this arch.")
    return {"loaded": matched, "inflated": inflated, "mismatched": mism, "missing": missing}


@torch.no_grad()
def _tta_logits(model, x):
    """Test-time augmentation: average logits over the D4 flip group
    {identity, hflip, vflip, rot180}. Exact (flips only — no interpolation), so
    every view is un-flipped before averaging. Steadier predictions / metric."""
    acc = None
    for dims in ((), (-1,), (-2,), (-2, -1)):
        xi = torch.flip(x, dims=dims) if dims else x
        li = model(xi)
        li = torch.flip(li, dims=dims) if dims else li
        acc = li if acc is None else acc + li
    return acc / 4.0


@torch.no_grad()
def _validate(model, loader, device, buffer: int, thr: float, tta: bool = False):
    """Headline IoU/Dice/Occlusion-Recall pooled over GLOBAL pixel counts
    (unbiased); buffered precision/recall/F1/IoU as per-batch means.
    With ``tta`` the model is evaluated under D4-flip test-time augmentation."""
    model.eval()
    tot = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "occ_tp": 0.0, "occ_total": 0.0}
    rel = {"relaxed_precision": 0.0, "relaxed_recall": 0.0, "relaxed_f1": 0.0, "relaxed_iou": 0.0}
    n = 0
    for batch in _pbar(loader, desc="  validating", unit="batch", leave=False):
        xb = batch["image"].to(device)
        logits = (_tta_logits(model, xb) if tta else model(xb)).float().cpu()
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


def _build_scheduler(opt, tr: dict, epochs: int):
    """Build an LR scheduler from ``cfg.train.scheduler``.

    Returns ``(scheduler, step_mode)`` where step_mode is:
      - ``"epoch"``   : call ``sched.step()`` once per epoch (cosine),
      - ``"plateau"`` : call ``sched.step(metric)`` after validation,
      - ``"none"``    : no scheduler (constant LR — unchanged legacy behaviour).

    Supported ``scheduler.name``: none | cosine (+ optional linear warmup) | plateau.
    Defaults to ``none`` when the block is absent, so existing configs are unaffected.
    """
    sc = tr.get("scheduler") or {}
    name = str(sc.get("name", "none")).lower()
    if name in ("none", "", "off", "constant"):
        return None, "none"
    min_lr = float(sc.get("min_lr", 1e-6))
    if name == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=float(sc.get("plateau_factor", 0.5)),
            patience=int(sc.get("plateau_patience", 5)), min_lr=min_lr)
        return sched, "plateau"
    if name == "cosine":
        warm = max(0, int(sc.get("warmup_epochs", 0)))
        base_lr = float(tr.get("lr", 1e-3))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs - warm), eta_min=min_lr)
        if warm > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=max(min_lr / base_lr, 1e-3), end_factor=1.0,
                total_iters=warm)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[warm])
        else:
            sched = cosine
        return sched, "epoch"
    raise ValueError(f"Unknown train.scheduler.name '{name}' (none|cosine|plateau)")


def _run_name(cfg: dict, stamp: str) -> str:
    """Self-documenting run-dir name: ``<arm>__<model_tag>__<stage>__<timestamp>``.

    arm   : vista | grove (cfg.arm.name) — keeps the two arms' runs separate.
    model : smp -> ``<decoder>-<encoder>``; else the arch/backbone name.
    stage : deepglobe→``pretrain``, tiles→``liss4``, synthetic→``synth``.
    e.g.  vista__segformer-mit_b2__pretrain__20260628_123245
          grove__ha_roadformer__liss4__20260628_150000

    Delegates to src/common/naming.py (the single source of truth for arm-aware
    artifact names, shared with predict.py and the GROVE tools).
    """
    from ...common.naming import run_name
    return run_name(cfg, stamp)


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
    init_from = tr.get("init_from")
    if init_from:                                    # warm-start from a pretrained ckpt
        load_pretrained(model, init_from, inflate_stem=bool(tr.get("init_inflate_stem", True)))
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
    tta = bool(cfg.get("eval", {}).get("tta", False))

    epochs = int(tr.get("epochs", 3))
    sched, sched_mode = _build_scheduler(opt, tr, epochs)
    es = tr.get("early_stop") or {}
    es_on = bool(es.get("enabled", False))
    es_patience = int(es.get("patience", 0))
    es_min_delta = float(es.get("min_delta", 0.0))
    print(f"[train] epochs={epochs} | scheduler={sched_mode if sched else 'none'} | "
          f"early_stop={'patience=' + str(es_patience) if es_on and es_patience > 0 else 'off'}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg["paths"]["runs"]) / "train" / _run_name(cfg, stamp)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = cfg.get("eval", {}).get("monitor", "occlusion_recall")
    best_val = -1.0
    best_ep = 0
    bad = 0                                              # epochs since last improvement

    for ep in range(1, epochs + 1):
        cur_lr = opt.param_groups[0]["lr"]
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

        val_metrics = _validate(model, val_loader, device, buffer, thr, tta=tta)
        row = {"epoch": ep, "lr": cur_lr,
               **{f"loss_{k}": v for k, v in ep_losses.items()}, **val_metrics}
        history.append(row)
        cur = val_metrics.get(best_key, -1)
        print(f"[ep {ep}/{epochs}] lr={cur_lr:.2e} loss={ep_losses['total']:.4f} "
              f"IoU={val_metrics['iou']:.3f} OccRec={val_metrics['occlusion_recall']:.3f} "
              f"relaxedF1={val_metrics['relaxed_f1']:.3f}")

        # checkpoint + early-stop bookkeeping (all monitored metrics are higher-better)
        if cur > best_val + es_min_delta:
            best_val, best_ep, bad = cur, ep, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                        "val": val_metrics}, out / "best.pt")
        else:
            bad += 1

        # LR schedule step (epoch-wise for cosine; metric-wise for plateau)
        if sched_mode == "epoch":
            sched.step()
        elif sched_mode == "plateau":
            sched.step(cur)

        if es_on and es_patience > 0 and bad >= es_patience:
            print(f"[train] early-stop: no {best_key} gain > {es_min_delta} in "
                  f"{es_patience} epochs (best={best_val:.3f} @ ep {best_ep}).")
            break

    pd.DataFrame(history).to_csv(out / "metrics.csv", index=False)

    # ---- figures: loss curve + qualitative prediction panel (paper artifacts) ----
    set_pub_style()
    import matplotlib.pyplot as plt
    hdf = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(hdf["epoch"], hdf["loss_total"], marker="o", label="total")
    ax.plot(hdf["epoch"], hdf["loss_cldice"], marker="s", label="clDice")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss")
    if "lr" in hdf:                                       # LR schedule on a twin axis
        axr = ax.twinx()
        axr.plot(hdf["epoch"], hdf["lr"], color="0.6", ls="--", lw=1, label="lr")
        axr.set_ylabel("lr"); axr.set_yscale("log")
    ax.legend(loc="upper right")
    save_fig(fig, fig_dir, "loss_curve")

    # qualitative panels for 3 RANDOM val patches; title/composite track the data source
    try:
        model.eval()
        src = _d.get("source", "synthetic")
        if src == "deepglobe":
            panel_title, rgb_order = "DeepGlobe (degraded RGB)", (1, 0, 2)   # R,G,B
        elif src == "tiles":
            panel_title, rgb_order = "LISS-IV FCC (NIR-R-G)", (2, 1, 0)      # CIR
        else:
            panel_title, rgb_order = "synthetic (false-color)", (2, 1, 0)
        n_panels = min(3, len(val_ds))
        pick = np.random.RandomState(int(rt.get("seed", 42)) + 1).choice(
            len(val_ds), size=n_panels, replace=False)
        with torch.no_grad():
            for j, idx in enumerate(pick):
                s = val_ds[int(idx)]
                logit = model(s["image"].unsqueeze(0).to(device)).float().cpu()[0]
                pname = "prediction" if j == 0 else f"prediction_{j + 1}"
                save_prediction_panel(s["image"], s["mask"], s["canopy"], logit,
                                      fig_dir, pname, thr=thr, title=panel_title,
                                      rgb_order=rgb_order)
    except Exception as e:  # noqa: BLE001 - artifact is best-effort
        print(f"[train] prediction panels skipped ({e})")

    print(f"[train] best {best_key}={best_val:.3f} @ epoch {best_ep} | artifacts -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase I training")
    ap.add_argument("--config", default="config/phase1/config.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
