"""Network performance — global efficiency (Latora & Marchiori, 2001).

Global efficiency = average of 1 / shortest-path-length over all node pairs. It
measures how efficiently the network moves traffic, and — unlike average path
length — stays well-defined when the graph FRAGMENTS (disconnected pair -> 1/inf
= 0). Exact is O(V·E); we SAMPLE source nodes for a fast estimate on big graphs.
"""
from __future__ import annotations

import random
from typing import Optional


def global_efficiency_sampled(graph, weight: Optional[str] = "travel_time_s",
                              n_samples: int = 300, seed: int = 42) -> float:
    """Approximate weighted global efficiency by sampling source nodes.

    weight=None => hop count (unweighted). Returns 0 for <2 nodes."""
    import networkx as nx
    nodes = list(graph.nodes())
    N = len(nodes)
    if N < 2:
        return 0.0
    srcs = nodes if N <= n_samples else random.Random(seed).sample(nodes, n_samples)
    total = 0.0
    for s in srcs:
        lengths = nx.single_source_dijkstra_path_length(graph, s, weight=weight)
        total += sum(1.0 / d for d in lengths.values() if d > 0)
    return total / (len(srcs) * (N - 1))
