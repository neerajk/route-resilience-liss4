"""Module A — Canopy characterisation.

Treepedia-style point sampling along the OSM street network + a per-segment
Overhead Canopy Occlusion Index (OCOI). This is the "bridge" layer that links
WHERE roads are hard to extract (under canopy) to WHERE the network is critical.

  sampling.py   -> systematic points along OSM edges (Treepedia sampling design)
  ocoi.py       -> per-segment OCOI from a CHM (+ optional NDVI) raster
  run_ocoi.py   -> CLI: OSM -> sample -> OCOI -> GeoJSON/CSV/figures
"""
