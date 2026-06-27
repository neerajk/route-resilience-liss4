"""Preprocessing: turn raw multi-sensor data into model-ready .npz tiles.

Stages (see pipeline.build_tiles):
  fetch -> reproject (EPSG:32643) -> NDVI -> co-register/resample CHM & S2 to the
  LISS-IV grid -> rasterise OSM -> tile -> save .npz (schema in data/dataset.py).

Modules:
  degrade.py     -> MTF/Gaussian blur-downsample (simulate coarser GSD)  [RUNS NOW]
  coregister.py  -> reproject/resample any raster onto a reference grid
  pipeline.py    -> orchestrator (skeleton; runs once you supply data + creds)
"""
