"""Betweenness centrality -> Gatekeeper nodes (bottlenecks).

Betweenness of a node = fraction of all shortest paths that pass through it
(Freeman 1977). High betweenness = a bottleneck the network depends on. Exact
betweenness is O(V·E); we k-SAMPLE source nodes for a fast approximation on large
city graphs (NetworkX `betweenness_centrality(k=...)`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def betweenness(graph, weight: Optional[str] = "travel_time_s",
                k: Optional[int] = 500, seed: int = 42) -> Dict:
    """Approximate (k-sampled) node betweenness, normalized to [0,1]."""
    import networkx as nx
    n = graph.number_of_nodes()
    kk = min(int(k), n) if k else None     # None => exact (slow)
    return nx.betweenness_centrality(graph, weight=weight, k=kk, seed=seed, normalized=True)


def gatekeepers(bc: Dict, top_n: int = 25) -> List[Tuple]:
    """Top-N nodes by betweenness (descending)."""
    return sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:int(top_n)]
