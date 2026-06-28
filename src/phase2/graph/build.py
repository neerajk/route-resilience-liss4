"""Step 4 — skeleton -> NetworkX graph via sknw (Image-Py/sknw, MIT).

sknw classifies skeleton pixels by neighbour count: 1 -> endpoint, >2 -> junction
(both become NODES); runs of 2-neighbour pixels become EDGES. Each node carries its
pixel coord `o`=[row,col]; each edge carries its pixel path `pts` ([[row,col],...])
and pixel `weight` (length). Coordinates are georeferenced in Step 5 (georef.py).

Ref: https://github.com/Image-Py/sknw  (used in SpaceNet/CRESI road pipelines).
"""
from __future__ import annotations

import numpy as np


def build_graph(skeleton: np.ndarray):
    """1-px skeleton -> NetworkX graph in PIXEL coordinates.

    Returns a networkx.Graph; nodes have attr 'o' (pixel [row,col]); edges have
    'pts' (pixel path) and 'weight' (pixel length)."""
    import sknw
    # multi=False: collapse parallel paths (simpler routable graph); ring=False
    # keeps isolated loops out. iso=False drops isolated single pixels (noise).
    graph = sknw.build_sknw(skeleton.astype(np.uint16), multi=False, iso=False, ring=False)
    return graph
