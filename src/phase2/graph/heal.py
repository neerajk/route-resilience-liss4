"""Step 6 — topological healing: reconnect occlusion-broken road fragments.

Occlusion (canopy/buildings) makes endpoints that are ACTUALLY connected appear
broken (PaRK-Detect / SpaceNet observation). We bridge dangling endpoints that are
(a) close (<= max_gap_m) and (b) well-aligned (<= max_angle_deg from the stub's
heading), choosing a MINIMAL, loop-free set via Union-Find + Kruskal's MST.
"""
from __future__ import annotations

import math

import numpy as np


class UnionFind:
    """Disjoint-Set (path compression + union by size). O(~1) find/union.
    Lets us add a bridge only between DIFFERENT components (no redundant loops)."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _stub_direction(graph, endpoint):
    """Outward unit heading of a degree-1 endpoint (world coords): from the
    adjacent geometry vertex toward the endpoint."""
    nbr = next(iter(graph[endpoint]))
    ex, ey = graph.nodes[endpoint]["x"], graph.nodes[endpoint]["y"]
    geom = graph[endpoint][nbr].get("geometry")
    if geom is not None and len(geom.coords) >= 2:
        c = list(geom.coords)
        d_start = (c[0][0] - ex) ** 2 + (c[0][1] - ey) ** 2
        d_end = (c[-1][0] - ex) ** 2 + (c[-1][1] - ey) ** 2
        x0, y0 = c[-2] if d_end <= d_start else c[1]   # vertex next to the endpoint
    else:
        x0, y0 = graph.nodes[nbr]["x"], graph.nodes[nbr]["y"]
    vx, vy = ex - x0, ey - y0
    n = math.hypot(vx, vy) or 1.0
    return vx / n, vy / n


def heal_graph(graph, max_gap_m: float = 60.0, max_angle_deg: float = 30.0,
               angle_penalty: float = 1.0):
    """Bridge dangling endpoints in place. Returns (graph, n_bridges)."""
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString

    endpoints = [n for n in graph.nodes if graph.degree(n) == 1]
    if len(endpoints) < 2:
        return graph, 0

    coords = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in endpoints])
    dirs = {n: _stub_direction(graph, n) for n in endpoints}
    tree = cKDTree(coords)
    max_angle = math.radians(max_angle_deg)

    # Union-Find seeded with existing connected components.
    uf = UnionFind(list(graph.nodes))
    for u, v in graph.edges():
        uf.union(u, v)

    # candidate bridges (i<j) passing distance + angle gates
    cands = []
    for i, j in tree.query_pairs(r=max_gap_m):
        ni, nj = endpoints[i], endpoints[j]
        if uf.find(ni) == uf.find(nj):
            continue                                   # already connected
        vx, vy = coords[j] - coords[i]
        dist = math.hypot(vx, vy)
        if dist < 1e-6:
            continue
        ux, uy = vx / dist, vy / dist
        dix, diy = dirs[ni]
        djx, djy = dirs[nj]
        ang_i = math.acos(max(-1.0, min(1.0, dix * ux + diy * uy)))
        ang_j = math.acos(max(-1.0, min(1.0, djx * -ux + djy * -uy)))
        ang = max(ang_i, ang_j)                        # both stubs must align
        if ang > max_angle:
            continue
        cost = dist * (1.0 + angle_penalty * (ang / max_angle))
        cands.append((cost, dist, ni, nj))

    # Kruskal: cheapest-first; add only if it joins two different components
    cands.sort(key=lambda t: t[0])
    n_bridges = 0
    for _cost, dist, ni, nj in cands:
        if uf.union(ni, nj):
            line = LineString([(graph.nodes[ni]["x"], graph.nodes[ni]["y"]),
                               (graph.nodes[nj]["x"], graph.nodes[nj]["y"])])
            graph.add_edge(ni, nj, healed=True, geometry=line, length_m=float(dist))
            n_bridges += 1
    return graph, n_bridges
