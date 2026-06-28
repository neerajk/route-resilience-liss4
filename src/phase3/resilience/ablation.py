"""Stress test — node ablation under different attack strategies.

Remove the top-k nodes by a strategy and re-measure efficiency: TARGETED
(betweenness) = simulate knocking out the most critical junctions (flood/closure);
DEGREE = busiest junctions; RANDOM = accidental failure. A network that collapses
fast under targeted removal is fragile (Albert, Jeong & Barabási, 2000).

Resilience Index = mean efficiency RETAINED across the removal sweep (area under
the curve), per strategy. 1 = unaffected, 0 = destroyed; lower = more vulnerable.
"""
from __future__ import annotations

import random
from typing import Dict, List

from . import centrality, efficiency


def node_order(graph, strategy: str, weight, k: int, seed: int) -> List:
    """Removal order (most-critical first) for a strategy. Computed once on the
    original graph (static attack)."""
    if strategy == "betweenness":
        bc = centrality.betweenness(graph, weight, k, seed)
        return [n for n, _ in sorted(bc.items(), key=lambda kv: kv[1], reverse=True)]
    if strategy == "degree":
        return [n for n, _ in sorted(graph.degree(), key=lambda kv: kv[1], reverse=True)]
    if strategy == "random":
        nodes = list(graph.nodes()); random.Random(seed).shuffle(nodes); return nodes
    raise ValueError(f"unknown strategy '{strategy}' (betweenness|degree|random)")


def ablate(graph, strategy: str, fractions: List[float], weight, base_eff: float,
           eff_samples: int, k: int, seed: int) -> Dict:
    """Sweep removal fractions; record efficiency retained + largest-CC fraction.

    Returns {fractions, efficiency_retained, lcc_fraction, resilience_index}."""
    import networkx as nx
    N = graph.number_of_nodes()
    order = node_order(graph, strategy, weight, k, seed)
    eff_ret, lcc_frac = [], []
    for f in fractions:
        H = graph.copy()
        H.remove_nodes_from(order[:int(f * N)])
        e = efficiency.global_efficiency_sampled(H, weight, eff_samples, seed) if H.number_of_nodes() > 1 else 0.0
        eff_ret.append(e / base_eff if base_eff > 0 else 0.0)
        lcc = max((len(c) for c in nx.connected_components(H)), default=0)
        lcc_frac.append(lcc / max(N, 1))
    resilience_index = sum(eff_ret) / len(eff_ret) if eff_ret else 0.0   # AUC ~ robustness
    return {"fractions": fractions, "efficiency_retained": eff_ret,
            "lcc_fraction": lcc_frac, "resilience_index": resilience_index}
