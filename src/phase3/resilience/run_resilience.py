"""Phase 3 orchestrator — road graph -> criticality + resilience stress test.

  load graph -> betweenness (Gatekeepers) -> baseline efficiency
             -> ablation (targeted vs degree vs random) -> Resilience Index

Config-driven (config/phase3/config_phase3.yaml -> resilience). Outputs:
runs/resilience/<ts>/ {criticality.geojson, gatekeepers.csv, resilience_curves.csv,
resilience_summary.csv, figures/resilience_curves}.

RUN:  python -m src.phase3.resilience.run_resilience --config config/phase3/config_phase3.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from ...common.config import load_config
from ...common.viz import save_fig, set_pub_style
from . import ablation, centrality, efficiency, io


def run(cfg: dict) -> Path:
    r = cfg.get("resilience", {})
    gpath = r.get("graph")
    if not gpath:
        raise RuntimeError("Set resilience.graph to a Phase 2 graph.graphml.")
    weight = r.get("weight", "travel_time_s")
    if weight in (None, "null", "none"):
        weight = None
    k = r.get("betweenness_k", 500)
    eff_n = int(r.get("efficiency_samples", 300))
    seed = int(r.get("seed", 42))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(r.get("out_dir", "runs/resilience")) / stamp
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[resilience] 1/5 load {gpath}")
    G = io.load_graph(gpath)
    crs = G.graph.get("crs")
    print(f"[resilience]     {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | weight={weight}")

    print(f"[resilience] 2/5 betweenness (k={k}) -> Gatekeepers")
    bc = centrality.betweenness(G, weight, k, seed)
    gks = centrality.gatekeepers(bc, int(r.get("gatekeepers_top_n", 25)))
    io.write_node_criticality(G, bc, crs, str(out / "criticality.geojson"))
    io.write_csv([[str(n), f"{v:.6f}", int(G.degree(n))] for n, v in gks],
                 ["node", "betweenness", "degree"], str(out / "gatekeepers.csv"))
    print(f"[resilience]     top Gatekeeper betweenness = {gks[0][1]:.4f}" if gks else "     (no nodes)")

    print(f"[resilience] 3/5 baseline efficiency (samples={eff_n})")
    base_eff = efficiency.global_efficiency_sampled(G, weight, eff_n, seed)

    ab = r.get("ablation", {})
    strategies = ab.get("strategies", ["betweenness", "degree", "random"])
    fr = np.linspace(0.0, float(ab.get("max_fraction", 0.10)), int(ab.get("steps", 11)))
    fractions = [float(x) for x in fr]

    print(f"[resilience] 4/5 ablation {strategies} up to {fractions[-1]:.0%} ...")
    curves, summary = {}, []
    for s in strategies:
        c = ablation.ablate(G, s, fractions, weight, base_eff, eff_n, k, seed)
        curves[s] = c
        summary.append([s, f"{c['resilience_index']:.4f}"])
        print(f"[resilience]     {s:<11} Resilience Index (mean eff. retained) = {c['resilience_index']:.3f}")

    print("[resilience] 5/5 export")
    rows = []
    for s, c in curves.items():
        for f, e, l in zip(c["fractions"], c["efficiency_retained"], c["lcc_fraction"]):
            rows.append([s, f"{f:.4f}", f"{e:.4f}", f"{l:.4f}"])
    io.write_csv(rows, ["strategy", "fraction_removed", "efficiency_retained", "lcc_fraction"],
                 str(out / "resilience_curves.csv"))
    io.write_csv(summary, ["strategy", "resilience_index"], str(out / "resilience_summary.csv"))

    set_pub_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s, c in curves.items():
        ax.plot([f * 100 for f in c["fractions"]], c["efficiency_retained"], marker="o", label=s)
    ax.set_xlabel("% nodes removed"); ax.set_ylabel("efficiency retained")
    ax.set_title("Network resilience under node ablation"); ax.legend()
    save_fig(fig, out / "figures", "resilience_curves")

    print(f"[resilience] DONE -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 — criticality + resilience (config-driven)")
    ap.add_argument("--config", default="config/phase3/config_phase3.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
