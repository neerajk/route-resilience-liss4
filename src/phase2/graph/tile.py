"""Tiled road-graph extraction for city-scale scenes.

sknw can't process the full scene at once, so we split it into NON-OVERLAPPING
blocks, build a georeferenced sub-graph per block (world coordinates), and UNION
them into one graph. Roads cut at block seams become dangling endpoints that the
GLOBAL heal step (heal.py) reconnects — so no overlap/dedup is needed. Reuses
binarize/skeleton/build/georef per block.
"""
from __future__ import annotations


def _pbar(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


def build_graph_tiled(arr, transform, crs, gcfg):
    """Full-scene mask -> unioned, georeferenced (pre-heal) NetworkX graph.

    Returns (graph, n_blocks, n_nonempty). Each block is binarized/skeletonized/
    built/georeferenced independently; sub-graphs are merged in world coords with
    globally-unique node ids. The seam gaps are left for the global heal."""
    import networkx as nx
    from rasterio.windows import Window, transform as win_transform

    from . import binarize, build, georef, skeleton

    t = gcfg.get("tiling", {})
    block = int(t.get("block_size", 2048))
    thr = float(gcfg.get("threshold", 0.5))
    minobj = int(gcfg.get("min_object_size", 50))
    closer = int(gcfg.get("closing_radius", 2))
    H, W = arr.shape

    G = nx.Graph(); G.graph["crs"] = str(crs)
    next_id = 0
    coords = [(r, c) for r in range(0, H, block) for c in range(0, W, block)]
    n_nonempty = 0
    for (r, c) in _pbar(coords, desc="blocks", unit="blk"):
        h, w = min(block, H - r), min(block, W - c)
        binb = binarize.clean_binary(arr[r:r + h, c:c + w], thr, minobj, closer)
        if not binb.any():
            continue
        g = build.build_graph(skeleton.skeletonize_mask(binb))
        if g.number_of_nodes() == 0:
            continue
        georef.georeference(g, win_transform(Window(c, r, w, h), transform), crs)
        # relabel to globally-unique ids, then merge into the running graph
        g = nx.relabel_nodes(g, {n: next_id + i for i, n in enumerate(g.nodes())})
        next_id += g.number_of_nodes()
        G.add_nodes_from(g.nodes(data=True))
        G.add_edges_from(g.edges(data=True))
        n_nonempty += 1
    return G, len(coords), n_nonempty
