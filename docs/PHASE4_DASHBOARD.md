# Phase 4 — Dashboard (tutorial + run)

**Task:** interactive web dashboard for the Route Resilience pipeline.
Visualises Phase 3 criticality results, resilience curves, and simulates network
failure from targeted node removal ("flood" or road closure).

Built with **Streamlit** (UI) + **Folium/Leaflet** (interactive map) + **Plotly**
(charts). Consumes Phase 3 output directories; requires no re-computation.

---

## Dependencies (one-time install into `rr` env)
The Phase 4 packages are in `environment.yml` (pip section). If you already have
the env, install them manually:
```bash
micromamba activate rr
pip install "streamlit>=1.32" "folium>=0.17" "streamlit-folium>=0.22" "plotly>=5.20"
```

---

## Setup

**1. Wire the Phase 3 run.**  
Phase 4 auto-discovers every directory under `runs/resilience/` that contains
`resilience_summary.csv`. If Phase 3 has been run at least once, the sidebar
shows a dropdown with all available runs. No config change needed.

**2. Enable the Flood Simulator (optional).**  
Edit `config/phase4/config_phase4.yaml` and set `graph_path` to the Phase 2
`graph.graphml` you want to stress-test:
```yaml
dashboard:
  graph_path: runs/graph/20240628_120000/graph.graphml
```
Leave it as `null` to keep the Flood Simulator tab disabled.

---

## Run (VS Code CLI)
Always run from the project root:
```bash
micromamba activate rr
streamlit run src/phase4/dashboard.py
```
The browser opens at `http://localhost:8501`. To use a custom config path:
```bash
RR_PHASE4_CONFIG=config/phase4/config_phase4.yaml streamlit run src/phase4/dashboard.py
```

---

## Tabs

### Tab 1 — Criticality Map
Interactive Leaflet map. Each node is a coloured circle:

| Colour | Meaning |
|---|---|
| Yellow | Low betweenness — low criticality |
| Orange | Medium betweenness |
| Red | High betweenness — **Gatekeeper** (critical junction) |

Circle **size** also scales with betweenness. Click any node for its ID,
betweenness score, and degree.

> Config: `max_map_nodes` caps how many nodes are rendered (default 5 000,
> top-N by betweenness). Raise it for detail, lower it for speed.

### Tab 2 — Resilience Curves
Plotly line chart: **global efficiency retained** vs **% nodes removed**, one
line per ablation strategy.

- **Betweenness (red)** — targeted attack: most critical junctions removed first.
- **Degree (orange)** — busiest junctions first.
- **Random (blue)** — accidental failure baseline.

A large gap between the red and blue curves means the network is fragile to
deliberate disruption. The **Resilience Index** table (below the chart) is the
area-under-curve per strategy; closer to 1.0 = more resilient.

### Tab 3 — Gatekeepers
Sortable table of the top-N critical junctions (default 25) with node ID,
betweenness score, and degree. High-betweenness, low-degree nodes are the most
dangerous single points of failure.

### Tab 4 — Flood Simulator
Requires `dashboard.graph_path` to be set (see Setup above).

1. **Select** one or more Gatekeeper nodes from the multiselect (pre-populated
   from `gatekeepers.csv`).
2. Click **Run simulation**.
3. The dashboard removes those nodes from the graph and reports:
   - Nodes removed
   - Change in connected components (splits = fragmentation)
   - Largest connected component before vs after
   - % connectivity lost (as fraction of LCC size)

A severity banner (green / orange / red) summarises the impact.

---

## Config reference (`config/phase4/config_phase4.yaml`)

| Key | Default | Description |
|---|---|---|
| `resilience_runs_dir` | `runs/resilience` | where to look for Phase 3 run dirs |
| `graph_path` | `null` | Phase 2 `graph.graphml`; enables Flood Simulator |
| `max_map_nodes` | `5000` | cap on nodes rendered on the Leaflet map |
