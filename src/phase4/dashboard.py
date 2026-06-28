"""Phase 4 — Route Resilience Dashboard (Streamlit + Folium/Leaflet + Plotly).

Consumes Phase 3 outputs from runs/resilience/<ts>/:
  criticality.geojson   node betweenness heatmap (points)
  gatekeepers.csv       top critical junctions
  resilience_curves.csv ablation efficiency decay curves
  resilience_summary.csv Resilience Index per strategy

Optional: Phase 2 graph.graphml (enables Flood Simulator tab).

RUN (from project root):
  streamlit run src/phase4/dashboard.py
  RR_PHASE4_CONFIG=config/phase4/config_phase4.yaml streamlit run src/phase4/dashboard.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# must be first streamlit call
st.set_page_config(
    page_title="Route Resilience",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── config ──────────────────────────────────────────────────────────────────

@st.cache_data
def _load_cfg() -> dict:
    import yaml
    path = os.environ.get("RR_PHASE4_CONFIG", "config/phase4/config_phase4.yaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ─── run discovery ───────────────────────────────────────────────────────────

def _discover_runs(runs_dir: str) -> list[str]:
    p = Path(runs_dir)
    if not p.exists():
        return []
    hits = sorted(p.glob("*/resilience_summary.csv"), reverse=True)
    return [str(h.parent) for h in hits]


# ─── data loaders ────────────────────────────────────────────────────────────

@st.cache_data
def _load_geojson(run_dir: str, max_nodes: int):
    import geopandas as gpd
    path = Path(run_dir) / "criticality.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if max_nodes and len(gdf) > max_nodes:
        gdf = gdf.nlargest(max_nodes, "betweenness").reset_index(drop=True)
    return gdf


@st.cache_data
def _load_csv(run_dir: str, fname: str) -> Optional[pd.DataFrame]:
    path = Path(run_dir) / fname
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def _load_graph(graph_path: str):
    import networkx as nx
    return nx.read_graphml(graph_path)


# ─── colour ramp (YlOrRd, no matplotlib dep) ─────────────────────────────────

_YLRD = [(255, 255, 178), (254, 204, 92), (253, 141, 60), (240, 59, 32), (189, 0, 38)]


def _bc_colour(val: float, vmin: float, vmax: float) -> str:
    norm = float(np.clip((val - vmin) / max(vmax - vmin, 1e-12), 0, 1))
    idx = norm * (len(_YLRD) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(_YLRD) - 1)
    t = idx - lo
    r = int(_YLRD[lo][0] * (1 - t) + _YLRD[hi][0] * t)
    g = int(_YLRD[lo][1] * (1 - t) + _YLRD[hi][1] * t)
    b = int(_YLRD[lo][2] * (1 - t) + _YLRD[hi][2] * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ─── tab 1: criticality map ──────────────────────────────────────────────────

def tab_map(gdf, max_nodes: int) -> None:
    import folium
    from streamlit_folium import st_folium

    if gdf is None or len(gdf) == 0:
        st.warning("criticality.geojson not found for this run.")
        return

    cx = float(gdf.geometry.x.mean())
    cy = float(gdf.geometry.y.mean())
    m = folium.Map(location=[cy, cx], zoom_start=12, tiles="CartoDB positron")

    vmin = float(gdf["betweenness"].min())
    vmax = float(gdf["betweenness"].max())

    rng = max(vmax - vmin, 1e-12)
    for _, row in gdf.iterrows():
        bc = float(row["betweenness"])
        colour = _bc_colour(bc, vmin, vmax)
        radius = 3 + 9 * (bc - vmin) / rng
        folium.CircleMarker(
            location=[float(row.geometry.y), float(row.geometry.x)],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            popup=folium.Popup(
                f"Node: {row['node']}<br>"
                f"Betweenness: {bc:.5f}<br>"
                f"Degree: {int(row['degree'])}",
                max_width=220,
            ),
            tooltip=f"BC={bc:.4f} deg={int(row['degree'])}",
        ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
         background:white;padding:10px 14px;border-radius:6px;
         font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,.3)">
      <b>Betweenness centrality</b><br>
      <span style="color:#ffff64;font-size:18px">&#9632;</span> Low (safe)<br>
      <span style="color:#fd8d3c;font-size:18px">&#9632;</span> Medium<br>
      <span style="color:#bd0026;font-size:18px">&#9632;</span> High (critical)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    note = f" (top {max_nodes:,} by betweenness)" if max_nodes and len(gdf) >= max_nodes else ""
    st.markdown(f"**Criticality heatmap** — node colour/size = betweenness{note}. Click a node for details.")
    st_folium(m, use_container_width=True, height=560, returned_objects=[])


# ─── tab 2: resilience curves ────────────────────────────────────────────────

def tab_curves(curves_df: Optional[pd.DataFrame], summary_df: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go

    if curves_df is None:
        st.warning("resilience_curves.csv not found for this run.")
        return

    _COLOURS = {"betweenness": "#d62728", "degree": "#ff7f0e", "random": "#1f77b4"}

    fig = go.Figure()
    for strategy, grp in curves_df.groupby("strategy"):
        grp = grp.sort_values("fraction_removed")
        fig.add_trace(go.Scatter(
            x=grp["fraction_removed"] * 100,
            y=grp["efficiency_retained"],
            mode="lines+markers",
            name=strategy,
            line=dict(color=_COLOURS.get(strategy, "#555"), width=2.5),
            marker=dict(size=7),
            hovertemplate=(
                f"<b>{strategy}</b><br>"
                "%{x:.1f}% removed<br>"
                "Efficiency retained: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Network resilience under node ablation",
        xaxis_title="% nodes removed",
        yaxis_title="Global efficiency retained",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(title="Attack strategy"),
        template="plotly_white",
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Betweenness** = targeted attack on the most critical junctions.  "
        "**Degree** = remove highest-connectivity nodes first.  "
        "**Random** = accidental failure.  "
        "A large gap between targeted and random curves = high vulnerability to deliberate disruption."
    )

    if summary_df is not None:
        st.markdown("---")
        st.markdown("**Resilience Index** — mean global efficiency retained over the removal sweep (higher = more resilient)")
        styled = summary_df.rename(
            columns={"strategy": "Strategy", "resilience_index": "Resilience Index"}
        )
        styled["Resilience Index"] = styled["Resilience Index"].apply(lambda v: f"{float(v):.4f}")
        st.dataframe(styled, use_container_width=False, hide_index=True)


# ─── tab 3: gatekeepers ──────────────────────────────────────────────────────

def tab_gatekeepers(gk_df: Optional[pd.DataFrame]) -> None:
    if gk_df is None:
        st.warning("gatekeepers.csv not found for this run.")
        return
    st.markdown(
        "**Top critical junctions** ranked by betweenness centrality.  "
        "These nodes control the most shortest-paths — losing one (flood, road closure) "
        "forces the largest network-wide detours."
    )
    display = gk_df.rename(columns={
        "node": "Node ID", "betweenness": "Betweenness", "degree": "Degree"
    }).copy()
    display["Betweenness"] = display["Betweenness"].apply(lambda v: f"{float(v):.6f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


# ─── tab 4: flood simulator ──────────────────────────────────────────────────

def tab_flood(gk_df: Optional[pd.DataFrame], graph_path: Optional[str]) -> None:
    if not graph_path or not Path(graph_path).exists():
        st.info(
            "Flood Simulator is disabled.  \n"
            "Set `dashboard.graph_path` in `config/phase4/config_phase4.yaml` "
            "to a Phase 2 `graph.graphml` to enable it."
        )
        st.code(
            "# config/phase4/config_phase4.yaml\n"
            "dashboard:\n"
            "  graph_path: runs/graph/<ts>/graph.graphml"
        )
        return

    G = _load_graph(graph_path)
    N, E = G.number_of_nodes(), G.number_of_edges()
    st.markdown(f"Graph loaded: **{N:,} nodes**, **{E:,} edges**")

    if gk_df is not None and len(gk_df):
        node_options = gk_df["node"].astype(str).tolist()
        label = "Select Gatekeeper nodes to flood / close:"
    else:
        node_options = list(str(n) for n in G.nodes())[:100]
        label = "Select nodes to remove (no gatekeepers.csv — showing first 100 nodes):"

    selected = st.multiselect(label, node_options, default=node_options[:1] if node_options else [])

    if not selected:
        st.info("Select at least one node above to simulate its removal.")
        return

    if st.button("Run simulation", type="primary"):
        import networkx as nx

        directed = G.is_directed()
        cc_fn = nx.weakly_connected_components if directed else nx.connected_components

        H = G.copy()
        removed = [n for n in selected if n in H]
        H.remove_nodes_from(removed)

        before_ccs = list(cc_fn(G))
        after_ccs = list(cc_fn(H))

        before_n_cc = len(before_ccs)
        after_n_cc = len(after_ccs)
        before_lcc = max((len(c) for c in before_ccs), default=0)
        after_lcc = max((len(c) for c in after_ccs), default=0)
        frac_lost = (before_lcc - after_lcc) / max(before_lcc, 1)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Nodes removed", len(removed))
        with c2:
            st.metric("Connected components", after_n_cc,
                      delta=after_n_cc - before_n_cc, delta_color="inverse")
        with c3:
            st.metric("Largest CC", f"{after_lcc:,}",
                      delta=f"{after_lcc - before_lcc:+,}", delta_color="inverse")
        with c4:
            st.metric("Connectivity lost", f"{frac_lost:.1%}")

        if frac_lost > 0.30:
            st.error(
                f"Critical failure — removing {', '.join(removed)} fragments "
                f"{frac_lost:.0%} of the network. Major detours or isolation expected."
            )
        elif frac_lost > 0.10:
            st.warning(
                f"Significant disruption — {frac_lost:.0%} connectivity lost. "
                "Secondary routes will be under heavy load."
            )
        else:
            st.success(
                f"Network remains largely intact — only {frac_lost:.1%} connectivity lost. "
                "Alternative routes can absorb the failure."
            )

        if after_n_cc > before_n_cc:
            st.markdown(
                f"The network split into **{after_n_cc} components** "
                f"(was {before_n_cc}). Isolated sub-networks have no path to the main graph."
            )


# ─── sidebar ─────────────────────────────────────────────────────────────────

def _sidebar(cfg: dict) -> tuple[Optional[str], Optional[str], int]:
    dcfg = cfg.get("dashboard", {})
    runs_dir = dcfg.get("resilience_runs_dir", "runs/resilience")
    graph_path = dcfg.get("graph_path") or None
    max_nodes = int(dcfg.get("max_map_nodes", 5000))

    st.sidebar.title("Route Resilience")
    st.sidebar.caption("Phase 4 — interactive dashboard")
    st.sidebar.markdown("---")

    runs = _discover_runs(runs_dir)
    if not runs:
        st.sidebar.warning(f"No Phase 3 runs found in `{runs_dir}/`.")
        st.sidebar.caption("Run Phase 3 first, then reload this page.")
        return None, graph_path, max_nodes

    labels = [Path(r).name for r in runs]
    choice = st.sidebar.selectbox("Phase 3 run", labels, index=0)
    run_dir = runs[labels.index(choice)]

    summary = _load_csv(run_dir, "resilience_summary.csv")
    if summary is not None:
        st.sidebar.markdown("**Resilience Index**")
        for _, row in summary.iterrows():
            st.sidebar.metric(str(row["strategy"]).capitalize(),
                              f"{float(row['resilience_index']):.3f}")

    st.sidebar.markdown("---")
    if graph_path and Path(graph_path).exists():
        st.sidebar.success("Flood Simulator: ON")
        st.sidebar.caption(f"`{graph_path}`")
    else:
        st.sidebar.info("Flood Simulator: set `graph_path` in Phase 4 config to enable.")

    return run_dir, graph_path, max_nodes


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_cfg()
    run_dir, graph_path, max_nodes = _sidebar(cfg)

    st.title("Route Resilience Dashboard")

    if run_dir is None:
        st.info("No Phase 3 runs found. Run Phase 3 then reload.")
        return

    gdf = _load_geojson(run_dir, max_nodes)
    curves_df = _load_csv(run_dir, "resilience_curves.csv")
    gk_df = _load_csv(run_dir, "gatekeepers.csv")
    summary_df = _load_csv(run_dir, "resilience_summary.csv")

    t1, t2, t3, t4 = st.tabs([
        "Criticality Map", "Resilience Curves", "Gatekeepers", "Flood Simulator"
    ])
    with t1:
        tab_map(gdf, max_nodes)
    with t2:
        tab_curves(curves_df, summary_df)
    with t3:
        tab_gatekeepers(gk_df)
    with t4:
        tab_flood(gk_df, graph_path)


if __name__ == "__main__":
    main()
