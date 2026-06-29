"""Exploratory Data Analysis (EDA) for Phase I — publication-grade.

WHY EDA FIRST (research practice): before modelling, quantify the data so the
paper can justify design choices with evidence rather than assertion.

  Q1  Class balance: what fraction of pixels are road? (motivates Dice/clDice;
      Milletari et al., 2016.)  — needs labels.
  Q2  Spectral separability: do canopy vs non-canopy (and road vs non-road, if
      labelled) differ in NDVI? (justifies the NDVI channel — Rouse et al., 1974.)
  Q3  Occlusion burden: what fraction of road pixels are UNDER canopy? — needs labels.
  Q4  Per-band statistics for normalisation (mean/std used at train time).

Works on BOTH:
  * synthetic tiles (labelled), and
  * real imagery-only LISS-IV tiles from ingest_liss4 (no road mask yet). In the
    label-less case the road questions (Q1, Q3, road-NDVI) are skipped and the
    NDVI evidence uses the CHM-derived CANOPY split instead — still a strong
    justification for the NDVI channel.

OUTPUTS (under cfg.paths.runs / 'eda/<timestamp>'):
  eda_report.md · band_statistics.csv · class_balance.csv · figures/*.{pdf,png}

Run:  python -m src.phase1.shared.eda.run_eda --config config/phase1/config.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import DEFAULT_CHANNELS, SyntheticRoadDataset, TileFolderDataset
from ....common.viz import save_fig, set_pub_style


def _load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_dataset(cfg: dict):
    d = cfg["data"]
    channels = tuple(d.get("channels", DEFAULT_CHANNELS))
    if d.get("source", "synthetic") == "synthetic":
        syn = d.get("synthetic", {})
        return SyntheticRoadDataset(
            length=int(d.get("eda_samples", 48)),
            size=int(d.get("tile_size", 256)),
            channels=channels,
            canopy_fraction=float(syn.get("canopy_fraction", 0.35)),
            n_roads=int(syn.get("n_roads", 9)),
            road_width=int(syn.get("road_width", 1)),
            seed=int(cfg.get("runtime", {}).get("seed", 42)),
        )
    # real LISS-IV tiles (ingest_liss4 output). NOTE: no norm => RAW stats.
    return TileFolderDataset(root=d["root"], channels=channels)


def _stretch(a, lo=2, hi=98):
    a = a.astype("float32")
    p_lo, p_hi = np.percentile(a, [lo, hi])
    return np.clip((a - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)


def run(cfg: dict) -> Path:
    set_pub_style()
    import matplotlib.pyplot as plt

    channels = list(cfg["data"].get("channels", DEFAULT_CHANNELS))
    ds = _build_dataset(cfg)

    # sample a subset so EDA is fast on large tile folders
    k = min(len(ds), int(cfg["data"].get("eda_samples", 48)))
    idxs = np.unique(np.linspace(0, len(ds) - 1, k).astype(int)).tolist()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg["paths"]["runs"]) / "eda" / stamp
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    gi = channels.index("green") if "green" in channels else 0
    ri = channels.index("red") if "red" in channels else 1
    ni = channels.index("nir") if "nir" in channels else 2
    ndvi_idx = channels.index("ndvi") if "ndvi" in channels else None

    per_channel = {c: [] for c in channels}
    road_frac, canopy_frac, occ_road_frac = [], [], []
    ndvi_canopy, ndvi_open, ndvi_road, ndvi_bg = [], [], [], []
    total_road = 0
    samples = []

    for i in idxs:
        s = ds[i]
        img = s["image"].numpy()                 # [C,H,W]
        mask = s["mask"].numpy()[0]
        canopy = s["canopy"].numpy()[0]
        for ci, c in enumerate(channels):
            per_channel[c].append(img[ci].ravel())
        road_frac.append(float(mask.mean()))
        canopy_frac.append(float(canopy.mean()))
        rd = (mask > 0.5)
        total_road += int(rd.sum())
        occ_road_frac.append(float(((rd) & (canopy > 0.5)).sum()) / float(max(rd.sum(), 1)))
        if ndvi_idx is not None:
            nd = img[ndvi_idx]
            ndvi_canopy.append(nd[canopy > 0.5]); ndvi_open.append(nd[canopy <= 0.5])
            if rd.any():
                ndvi_road.append(nd[rd]); ndvi_bg.append(nd[~rd])
        if len(samples) < 6:
            samples.append((img, mask, canopy))

    has_labels = total_road > 0

    # --- band statistics (Q4) --------------------------------------------- #
    rows = []
    for c in channels:
        vals = np.concatenate(per_channel[c])
        rows.append({"channel": c, "mean": float(vals.mean()), "std": float(vals.std()),
                     "min": float(vals.min()), "p01": float(np.percentile(vals, 1)),
                     "p50": float(np.percentile(vals, 50)), "p99": float(np.percentile(vals, 99)),
                     "max": float(vals.max())})
    band_df = pd.DataFrame(rows)
    band_df.to_csv(out / "band_statistics.csv", index=False)

    cb_metrics = [("canopy_fraction", canopy_frac)]
    if has_labels:
        cb_metrics = [("road_pixel_fraction", road_frac), ("canopy_fraction", canopy_frac),
                      ("occluded_road_fraction", occ_road_frac)]
    cb = pd.DataFrame({"metric": [m for m, _ in cb_metrics],
                       "mean": [float(np.mean(v)) for _, v in cb_metrics],
                       "std": [float(np.std(v)) for _, v in cb_metrics]})
    cb.to_csv(out / "class_balance.csv", index=False)

    # --- FIGURE 1: per-channel histograms (normalisation sanity) ---------- #
    fig, axes = plt.subplots(1, len(channels), figsize=(3.2 * len(channels), 3))
    if len(channels) == 1:
        axes = [axes]
    for ax, c in zip(axes, channels):
        ax.hist(np.concatenate(per_channel[c]), bins=60); ax.set_title(c); ax.set_yticks([])
    fig.suptitle("Per-channel value distributions (raw)")
    save_fig(fig, fig_dir, "01_channel_histograms")

    # --- FIGURE 2: NDVI separability (canopy split always; road split if labelled) #
    if ndvi_idx is not None and ndvi_canopy:
        fig, ax = plt.subplots(1, 2 if has_labels else 1, figsize=(10 if has_labels else 5, 4),
                               squeeze=False)
        a0 = ax[0][0]
        a0.hist(np.concatenate(ndvi_open), bins=60, alpha=0.6, density=True, label="non-canopy")
        a0.hist(np.concatenate(ndvi_canopy), bins=60, alpha=0.6, density=True, label="canopy")
        a0.set_xlabel("NDVI"); a0.set_ylabel("density")
        a0.set_title("NDVI: canopy vs non-canopy"); a0.legend()
        if has_labels:
            a1 = ax[0][1]
            a1.hist(np.concatenate(ndvi_bg), bins=60, alpha=0.6, density=True, label="non-road")
            a1.hist(np.concatenate(ndvi_road), bins=60, alpha=0.6, density=True, label="road")
            a1.set_xlabel("NDVI"); a1.set_title("NDVI: road vs non-road"); a1.legend()
        save_fig(fig, fig_dir, "02_ndvi_separability")

    # --- FIGURE 3: sample tile stack (FCC + channels + masks) ------------- #
    img, mask, canopy = samples[0]
    fcc = np.dstack([_stretch(img[ni]), _stretch(img[ri]), _stretch(img[gi])])
    panels = [("FCC (NIR-R-G)", fcc, None)]
    for ci, c in enumerate(channels):
        panels.append((c, img[ci], "viridis"))
    panels.append(("canopy", canopy, "Greens"))
    if has_labels:
        panels.append(("road mask", mask, "gray"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 2.8))
    for ax, (title, arr, cmap) in zip(axes, panels):
        ax.imshow(arr) if cmap is None else ax.imshow(arr, cmap=cmap)
        ax.set_title(title); ax.axis("off")
    save_fig(fig, fig_dir, "03_sample_tile")

    # --- FIGURE 4: FCC contact sheet --------------------------------------- #
    nfc = len(samples)
    fig, axes = plt.subplots(1, nfc, figsize=(2.4 * nfc, 2.6))
    if nfc == 1:
        axes = [axes]
    for ax, (im, _, _) in zip(axes, samples):
        ax.imshow(np.dstack([_stretch(im[ni]), _stretch(im[ri]), _stretch(im[gi])]))
        ax.axis("off")
    fig.suptitle("Sample tiles — false-color (NIR-R-G)")
    save_fig(fig, fig_dir, "04_fcc_contact_sheet")

    # --- markdown report --------------------------------------------------- #
    try:
        band_table = band_df.to_markdown(index=False)
    except ImportError:
        band_table = "```\n" + band_df.to_string(index=False) + "\n```"
    src = cfg["data"].get("source", "synthetic")
    lines = [
        "# Phase I EDA report\n",
        f"- Generated: {stamp}",
        f"- Data source: `{src}` | tiles analysed: {len(idxs)} of {len(ds)}",
        f"- Channels: {channels}",
        f"- Labels present: **{has_labels}**\n",
        "## Class balance",
        f"- Canopy fraction: **{np.mean(canopy_frac):.3f}** (±{np.std(canopy_frac):.3f}) "
        "— from CHM > threshold (the occlusion burden).",
    ]
    if has_labels:
        lines += [
            f"- Road pixel fraction: **{np.mean(road_frac):.4f}** (±{np.std(road_frac):.4f}) "
            "-> strong imbalance motivates Dice/clDice.",
            f"- **Occluded-road fraction: {np.mean(occ_road_frac):.3f}** "
            "-> share of roads under canopy (the problem we solve).",
        ]
    else:
        lines += ["- Road labels NOT present (imagery-only) -> road/occlusion stats skipped. "
                  "Add labels (OSM/vector, write_mask=true) to enable Q1/Q3."]
    lines += [
        "\n## Band statistics (for cfg.data.norm)",
        band_table,
        "\n## Figures",
        "- `01_channel_histograms` — dynamic range / normalisation sanity.",
        "- `02_ndvi_separability` — NDVI separates canopy vs non-canopy"
        + (" (and road vs non-road)." if has_labels else " (justifies the NDVI channel)."),
        "- `03_sample_tile` — qualitative look at the input stack.",
        "- `04_fcc_contact_sheet` — false-color overview of sample tiles.\n",
        "## References",
        "- Rouse et al. (1974) NDVI; Milletari et al. (2016) Dice; Shit et al. (2021) clDice.",
    ]
    (out / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[EDA] source={src} labels={has_labels} | "
          f"report + {len(list(fig_dir.glob('*.png')))} figures -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase I EDA")
    ap.add_argument("--config", default="config/phase1/config.yaml")
    args = ap.parse_args()
    run(_load_config(args.config))


if __name__ == "__main__":
    main()
