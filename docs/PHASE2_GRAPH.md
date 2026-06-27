# Phase 2 — Road Network Extraction (tutorial / reference)

**Task name:** *Road Network Extraction* (a.k.a. road graph extraction / road
topology extraction). Converts a road **raster mask** → **routable vector graph**,
then **heals** occlusion-broken gaps. Feeds Phase 3 (criticality/resilience).

**Roles assumed while coding:** GeoAI researcher · remote-sensing/geoinformatics
engineer · data scientist · GIS analyst · AI/ML researcher · AI Architect.
Conventions: config-driven, graceful degradation (lazy imports), clear comments +
inline references, correct geospatial handling (CRS, metre distances, transforms).

## External code referenced (attribution)
- **Image-Py/sknw** (MIT) — `build_sknw`: skeleton image → NetworkX graph. *Dependency.*
- **CosmiQ/apls** (Apache-2.0) — APLS graph-similarity metric. *Reference for topo-accuracy.*
- **avanetten/cresi** (Apache-2.0) — mask→sknw→graph→speeds pipeline. *Pattern reference.*
- **scikit-image** — `skeletonize`, morphology. **NetworkX** — graph + (Phase 3) centrality.

## Pipeline (input → logic → output)

| Step | Module | Logic | Out |
|---|---|---|---|
| 1 Read | `io.read_mask` | rasterio → (array, **transform**, **CRS**) — the georef that keeps the graph in real coords | array+geo |
| 2 Clean | `binarize.clean_binary` | threshold → `remove_small_objects` (drop specks=fake nodes) → `binary_closing` (bridge 1–2 px) | binary |
| 3 Skeleton | `skeleton.skeletonize_mask` | morphological thinning → 1-px centerlines (road "spine") | skeleton |
| 4 Build | `build.build_graph` | `sknw`: pixels with 1 neighbour=endpoint, >2=junction (nodes); 2=path (edges) | graph (pixel) |
| 5 Georef | `georef.georeference` | affine transform: `x,y = T*(col+0.5,row+0.5)`; LineString per edge; length in metres | graph (world) |
| 6 Heal | `heal.heal_graph` | endpoints (deg-1) → KDTree neighbours ≤ `max_gap_m` → angle gate ≤ `max_angle_deg` → cost=dist×angle → **Union-Find + Kruskal MST** bridge only *different* components | connected graph |
| 7 Weight | `weights.add_weights` | edge length (m) → travel-time via `speed_kph_default` | weighted |
| 8 Metrics | `metrics` | **Connectivity Ratio** = largest-CC after/before healing; node/edge/component counts | metrics.csv |
| 9 Export | `run_graph` | `graph.graphml` (→ Phase 3) · `roads.geojson` (QGIS) · overlay PNG | artifacts |

## Healing logic (the core)
- **Union-Find (Disjoint-Set):** O(1) "same group?" / "merge". Pre-union all existing
  edges → each road fragment = one group. A candidate bridge is added only if its two
  endpoints are in **different** groups (else it's a redundant loop).
- **Kruskal MST:** sort candidate bridges by `cost = dist·(1 + penalty·angle)`,
  add cheapest-first when it joins two groups → **minimal** reconnection, no cycles.
- **Angle gate:** a real road continues roughly straight; reject a bridge whose
  direction deviates > `max_angle_deg` from the dangling stub's heading.
- **Why heal at all:** occlusion (canopy/buildings) makes endpoints that are *actually*
  connected look broken (PaRK-Detect/SpaceNet observation).

## Inputs
- **Contract:** a road-mask **GeoTIFF** — `pred_mask.tif` (model) OR an OSM mask (dev).
- Dev helper `make_osm_mask.py` builds an OSM road-mask GeoTIFF on a reference grid.

## Config (`config/phase2/config_phase2.yaml → graph`)
`mask · threshold · min_object_size · closing_radius · heal:{max_gap_m, max_angle_deg, angle_penalty} · speed_kph_default · out_dir`

## Run
```bash
python -m src.phase2.graph.run_graph --config config/phase2/config_phase2.yaml
```
