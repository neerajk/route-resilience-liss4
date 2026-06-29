"""VISTA-v2 publishable PE-comparison plots.

Reads the per-variant runs (same glob as bench.py) and renders publication-style
figures (PNG + PDF) via the project's viz styling:

  1. grouped bar — OccRec / IoU / relaxed-F1 by PE, error bars = 95% CI
  2. paired per-fold lines — each fold's OccRec across the 4 variants
  3. training-curve overlay — val OccRec vs epoch, one line per PE
  4. params/efficiency context (optional, if best.pt present)

RUN:  python -m src.phase1.vista_v2.plots --runs "runs/train/vista_v2-*_liss4_*" --out runs/vista_v2_bench/figures
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .bench import _best_row, _ci95, _variant_of


def _gather(run_glob: str):
    per = {}
    curves = {}
    for r in sorted(glob.glob(run_glob)):
        run = Path(r)
        row = _best_row(run)
        if row is None:
            continue
        v = _variant_of(run)
        per.setdefault(v, []).append(row)
        mcsv = run / "metrics.csv"
        if mcsv.exists():
            curves.setdefault(v, []).append(pd.read_csv(mcsv))
    return per, curves


def render(run_glob: str, out_dir: str):
    import matplotlib.pyplot as plt
    try:
        from ...common.viz import save_fig, set_pub_style
        set_pub_style()
    except Exception:
        def save_fig(fig, d, name):
            Path(d).mkdir(parents=True, exist_ok=True)
            for ext in ("png", "pdf"):
                fig.savefig(Path(d) / f"{name}.{ext}", bbox_inches="tight", dpi=200)

    per, curves = _gather(run_glob)
    out = Path(out_dir)
    order = [v for v in ["botnet", "rope", "sincos", "nope"] if v in per] or list(per)

    # 1. grouped bar with 95% CI
    metrics = ["occlusion_recall", "iou", "relaxed_f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics)); width = 0.8 / max(len(order), 1)
    for i, v in enumerate(order):
        d = pd.DataFrame(per[v]); means, errs = [], []
        for mt in metrics:
            vals = d[mt].values if mt in d.columns else np.array([np.nan])
            lo, hi = _ci95(vals)
            means.append(np.nanmean(vals)); errs.append((hi - lo) / 2 if not np.isnan(lo) else 0)
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=v)
    ax.set_xticks(x + width * (len(order) - 1) / 2)
    ax.set_xticklabels(["Occlusion-Recall", "IoU", "relaxed-F1"])
    ax.set_ylabel("score"); ax.set_title("VISTA-v2 — PE comparison (mean ± 95% CI)")
    ax.legend(title="PE")
    save_fig(fig, out, "pe_bar_ci"); plt.close(fig)

    # 2. paired per-fold OccRec lines
    fig, ax = plt.subplots(figsize=(6, 4))
    nfold = min((len(per[v]) for v in order), default=0)
    if nfold >= 2:
        for f in range(nfold):
            ys = [pd.DataFrame(per[v]).iloc[f]["occlusion_recall"] for v in order]
            ax.plot(order, ys, marker="o", alpha=0.6, label=f"fold {f}")
        ax.set_ylabel("Occlusion-Recall"); ax.set_title("Per-fold OccRec across PE")
        ax.legend(fontsize=7, ncol=2)
    save_fig(fig, out, "pe_paired_folds"); plt.close(fig)

    # 3. training-curve overlay (mean across folds)
    fig, ax = plt.subplots(figsize=(6, 4))
    for v in order:
        if v in curves:
            m = pd.concat(curves[v]).groupby("epoch")["occlusion_recall"].mean()
            ax.plot(m.index, m.values, label=v)
    ax.set_xlabel("epoch"); ax.set_ylabel("val Occlusion-Recall")
    ax.set_title("Training curves by PE"); ax.legend(title="PE")
    save_fig(fig, out, "pe_training_curves"); plt.close(fig)

    print(f"[plots] -> {out}/ (pe_bar_ci · pe_paired_folds · pe_training_curves)  PNG+PDF")


def main() -> None:
    ap = argparse.ArgumentParser(description="VISTA-v2 PE plots")
    ap.add_argument("--runs", default="runs/train/vista_v2-*_liss4_*")
    ap.add_argument("--out", default="runs/vista_v2_bench/figures")
    args = ap.parse_args()
    render(args.runs, args.out)


if __name__ == "__main__":
    main()
