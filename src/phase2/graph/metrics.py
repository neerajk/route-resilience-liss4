"""Step 8 — graph metrics. Headline: Connectivity Ratio (healing effectiveness).

Connectivity Ratio = size of the largest connected component AFTER healing / BEFORE.
> 1 means healing merged fragments into bigger routable components. Also reports
node/edge/component counts + total road length. (Topo-accuracy vs OSM via APLS,
CosmiQ/apls, is a future addition.)
"""
from __future__ import annotations

from typing import Dict


def graph_stats(graph) -> Dict[str, float]:
    """Basic structure stats incl. largest connected component (by node count)."""
    import networkx as nx
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    comps = list(nx.connected_components(graph)) if n_nodes else []
    lcc = max((len(c) for c in comps), default=0)
    total_len = sum(float(d.get("length_m", 0.0)) for _, _, d in graph.edges(data=True))
    return {"n_nodes": n_nodes, "n_edges": n_edges, "n_components": len(comps),
            "largest_cc_nodes": lcc, "total_length_km": total_len / 1000.0}


def connectivity_ratio(lcc_before: int, lcc_after: int, eps: float = 1e-9) -> float:
    """Largest-CC growth from healing (the headline Phase-2 number)."""
    return (lcc_after + eps) / (lcc_before + eps)
