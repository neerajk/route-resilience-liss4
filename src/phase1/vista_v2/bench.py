"""VISTA-v2 PE benchmark + statistics.

Collates the per-variant runs (botnet | rope | sincos | nope), reading each run's
metrics.csv (best epoch by Occlusion-Recall) and best.pt cfg (to identify the PE
variant robustly, regardless of dir name). With multiple runs per variant (one per
spatial-block fold), it runs PAIRED statistics across folds:

  - mean ± 95% CI (t-based) per variant per metric
  - paired Wilcoxon signed-rank vs the default (botnet) on per-fold OccRec
  - Holm-corrected p-values (multiple comparisons) + Cohen's d effect size

HONEST CAVEAT (printed): with ~7 folds, power is low — lead with effect sizes + CIs,
treat p-values as indicative. Writes benchmark.csv + stats.json for plots.py.

RUN:  python -m src.phase1.vista_v2.bench --runs "runs/train/*vista_v2-*" --out runs/vista_v2_bench
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["occlusion_recall", "iou", "dice", "relaxed_f1"]


def _variant_of(run: Path) -> str:
    try:
        import torch
        ck = torch.load(run / "best.pt", map_location="cpu")
        return str((ck.get("cfg", {}).get("model", {}).get("pe", {}) or {}).get("type", "?"))
    except Exception:
        # fall back to the dir name tag vista_v2-<pe>_...
        name = run.name
        return name.split("vista_v2-")[-1].split("_")[0] if "vista_v2-" in name else "?"


def _best_row(run: Path):
    mcsv = run / "metrics.csv"
    if not mcsv.exists():
        return None
    df = pd.read_csv(mcsv)
    key = "occlusion_recall" if "occlusion_recall" in df.columns else df.columns[-1]
    return df.loc[df[key].idxmax()]


def _ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m, s = x.mean(), x.std(ddof=1)
    h = 1.96 * s / np.sqrt(len(x))
    return (m - h, m + h)


def _cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def _holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run_max = 0.0
    for rank, idx in enumerate(order):
        v = (len(pvals) - rank) * pvals[idx]
        run_max = max(run_max, min(v, 1.0))
        adj[idx] = run_max
    return adj


def collate(run_glob: str, out_dir: str, default: str = "botnet"):
    runs = [Path(r) for r in sorted(glob.glob(run_glob))]
    per_variant = {}
    for r in runs:
        row = _best_row(r)
        if row is None:
            continue
        v = _variant_of(r)
        per_variant.setdefault(v, []).append(row)

    # descriptive table
    rows = []
    for v, recs in per_variant.items():
        d = pd.DataFrame(recs)
        entry = {"pe": v, "n_runs": len(recs)}
        for mt in METRICS:
            if mt in d.columns:
                lo, hi = _ci95(d[mt].values)
                entry[f"{mt}_mean"] = round(float(d[mt].mean()), 4)
                entry[f"{mt}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(entry)
    table = pd.DataFrame(rows).sort_values("occlusion_recall_mean", ascending=False)

    # paired stats vs default on OccRec
    stats = {"default": default, "metric": "occlusion_recall", "comparisons": []}
    if default in per_variant:
        base = pd.DataFrame(per_variant[default])["occlusion_recall"].values
        comps, praw = [], []
        for v, recs in per_variant.items():
            if v == default:
                continue
            arr = pd.DataFrame(recs)["occlusion_recall"].values
            p = float("nan")
            if len(arr) == len(base) and len(base) >= 3:
                try:
                    from scipy.stats import wilcoxon
                    p = float(wilcoxon(base, arr).pvalue)
                except Exception:
                    p = float("nan")
            comps.append({"pe": v, "n": len(arr), "p_raw": p,
                          "cohens_d": _cohen_d(base, arr)})
            praw.append(p)
        adj = _holm(np.array([c["p_raw"] if not np.isnan(c["p_raw"]) else 1.0 for c in comps])) \
            if comps else []
        for c, a in zip(comps, adj):
            c["p_holm"] = round(float(a), 4)
        stats["comparisons"] = comps

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "benchmark.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(table.to_markdown(index=False))
    print("\n[stats] paired Wilcoxon vs", default, "(OccRec); Holm-corrected:")
    for c in stats["comparisons"]:
        print(f"  {c['pe']:8s} n={c['n']} p_raw={c['p_raw']:.3f} "
              f"p_holm={c.get('p_holm', float('nan'))} d={c['cohens_d']:.2f}")
    print("[caveat] ~7 folds → low power; lead with effect sizes + CIs, not p-values.")
    print(f"[bench] -> {out}/benchmark.csv · stats.json")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="VISTA-v2 PE benchmark + stats")
    ap.add_argument("--runs", default="runs/train/*vista_v2-*")
    ap.add_argument("--out", default="runs/vista_v2_bench")
    ap.add_argument("--default", default="botnet")
    args = ap.parse_args()
    collate(args.runs, args.out, args.default)


if __name__ == "__main__":
    main()
