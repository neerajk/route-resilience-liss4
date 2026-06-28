# Phase 3 — Network Analysis & Stress Testing (tutorial + run)

**Task:** *road-network resilience analysis* — find the critical junctions and
quantify how the network degrades when they fail. Consumes Phase 2's
`graph.graphml`; feeds Phase 4 (dashboard).

**Roles assumed:** GeoAI researcher · GIS analyst · data scientist · network
scientist · AI Architect. Config-driven, graceful degradation, cited methods.

## References (methods)
- **Freeman (1977)** — betweenness centrality (bottleneck identification).
- **Latora & Marchiori (2001)** — global efficiency (network performance).
- **Albert, Jeong & Barabási (2000)** — error vs attack tolerance (targeted removal
  collapses fragile networks fastest). **NetworkX** for all graph algorithms.

## Pipeline (input → logic → output)

| Step | Module | Logic | Out |
|---|---|---|---|
| 1 Load | `io.load_graph` | read `graph.graphml` → NetworkX; coerce numeric attrs | graph |
| 2 Betweenness | `centrality.betweenness` | fraction of shortest paths through each node (Freeman); **k-sampled** for speed → **Gatekeepers** | bc dict |
| 3 Baseline | `efficiency.global_efficiency_sampled` | mean 1/shortest-path (Latora–Marchiori), sampled; robust to fragmentation | base efficiency |
| 4 Ablation | `ablation.ablate` | remove top-k nodes by **targeted / degree / random**; re-measure efficiency + largest-CC at each step | decay curves |
| 5 Index | `ablation` + `run_resilience` | **Resilience Index** = mean efficiency retained over the sweep (per strategy) | summary + plot |

## Key logic
- **Betweenness (Gatekeepers):** a node on many shortest paths is a bottleneck —
  losing it (flood/closure) forces long detours. k-sampling approximates it fast.
- **Global efficiency:** unlike average path length, it stays defined when the graph
  splits (disconnected pair → 1/∞ = 0), so it's the right ablation measure.
- **Targeted vs random:** removing high-betweenness nodes first = a smart attack;
  a fragile network's efficiency collapses far faster than under random failure —
  the gap between the curves is the vulnerability story.
- **Resilience Index** ∈ [0,1]: 1 = unaffected, lower = more vulnerable.

## Outputs (`runs/resilience/<ts>/`)
- `criticality.geojson` — nodes coloured by betweenness (QGIS heatmap)
- `gatekeepers.csv` — top-N critical junctions
- `resilience_curves.csv` — efficiency retained + LCC fraction vs % removed, per strategy
- `resilience_summary.csv` — Resilience Index per strategy
- `figures/resilience_curves.{pdf,png}` — the decay curves

## Run (VS Code CLI)
```bash
micromamba activate rr
# 1. point the config at a Phase 2 graph:
#    config/phase3/config_phase3.yaml -> resilience.graph: runs/graph/<ts>/graph.graphml
python -m src.phase3.resilience.run_resilience --config config/phase3/config_phase3.yaml
```
**Big graph & slow?** lower `efficiency_samples` (e.g. 100) and `betweenness_k`
(e.g. 200), or reduce `ablation.steps`. CPU only — no GPU needed.
