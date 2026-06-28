"""Step 7 — edge weights for routing/criticality.

length_m comes from georef (planar metres). travel_time_s = length / speed. Real
OSM road classes -> speeds can be added later; model-derived edges use a default.
"""
from __future__ import annotations


def add_weights(graph, speed_kph_default: float = 30.0):
    """Annotate edges with travel_time_s (from length_m + a default speed)."""
    mps = float(speed_kph_default) * 1000.0 / 3600.0    # km/h -> m/s
    for _u, _v, d in graph.edges(data=True):
        length = float(d.get("length_m", 0.0))
        d["speed_kph"] = float(speed_kph_default)
        d["travel_time_s"] = length / mps if mps > 0 else 0.0
    return graph
