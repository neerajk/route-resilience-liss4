"""Backbone benchmark aggregator — collate VISTA / CSWin / HA-RoadFormer runs.

Reads each run's metrics.csv (best epoch by occlusion_recall) and best.pt (param
count), parses the arm/backbone from the run-dir name (naming.py scheme), and emits
one comparison table (markdown + csv). This is the Stage-7 mask-level comparison:
the clean, graph-free VISTA-vs-GROVE-backbone numbers.

RUN:  python -m src.phase1.grove.bench --runs "runs/train/*liss4*" --out runs/bench
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def _params(run: Path) -> float:
    ckpt = run / "best.pt"
    if not ckpt.exists():
        return float("nan")
    try:
        import torch
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd)
        return sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel")) / 1e6
    except Exception:
        return float("nan")


def collate(run_globs, out_dir: str) -> pd.DataFrame:
    runs = []
    for g in run_globs:
        runs.extend(sorted(glob.glob(g)))
    rows = []
    for r in runs:
        run = Path(r)
        mcsv = run / "metrics.csv"
        if not mcsv.exists():
            continue
        df = pd.read_csv(mcsv)
        key = "occlusion_recall" if "occlusion_recall" in df.columns else df.columns[-1]
        best = df.loc[df[key].idxmax()]
        parts = run.name.split("__")           # <arm>__<model>__<stage>__<stamp>
        rows.append({
            "run": run.name,
            "arm": parts[0] if parts else "?",
            "backbone": parts[1] if len(parts) > 1 else "?",
            "occlusion_recall": round(float(best.get("occlusion_recall", float("nan"))), 4),
            "iou": round(float(best.get("iou", float("nan"))), 4),
            "dice": round(float(best.get("dice", float("nan"))), 4),
            "best_epoch": int(best.get("epoch", -1)),
            "params_M": round(_params(run), 2),
        })
    table = pd.DataFrame(rows).sort_values("occlusion_recall", ascending=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "benchmark.csv", index=False)
    (out / "benchmark.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    print(table.to_markdown(index=False))
    print(f"\n[bench] -> {out}/benchmark.{{csv,md}}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="GROVE backbone benchmark aggregator")
    ap.add_argument("--runs", nargs="+", required=True, help="glob(s) of run dirs")
    ap.add_argument("--out", default="runs/bench")
    args = ap.parse_args()
    collate(args.runs, args.out)


if __name__ == "__main__":
    main()
